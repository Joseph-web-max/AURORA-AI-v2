"""AURORA controlled evolution.

feedback -> admin review -> reviewed dataset -> evaluation -> fine-tune -> gated activation.
Nothing a user says changes the model or the dataset until an admin (AURORA_ADMIN_EMAILS) approves it,
and a fine-tuned model is only activatable if it beats the base model on held-out reviewed examples.
"""
import hmac
import json
import os
import re
import secrets
import time

import httpx
from fastapi import Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import text

DDL = [
    "CREATE TABLE IF NOT EXISTS evo_feedback (id TEXT PRIMARY KEY, user_id BIGINT, prompt TEXT, answer TEXT, rating INT, correction TEXT, status TEXT, ts BIGINT)",
    "CREATE TABLE IF NOT EXISTS evo_data (id TEXT PRIMARY KEY, prompt TEXT, answer TEXT, tag TEXT, holdout INT, ts BIGINT)",
    "CREATE TABLE IF NOT EXISTS evo_gaps (id TEXT PRIMARY KEY, gkey TEXT, prompt TEXT, n INT, status TEXT, ts BIGINT)",
    "CREATE TABLE IF NOT EXISTS evo_models (id TEXT PRIMARY KEY, name TEXT, tag TEXT, job TEXT, status TEXT, score REAL, base_score REAL, ts BIGINT)",
    "CREATE TABLE IF NOT EXISTS evo_meta (k TEXT PRIMARY KEY, v TEXT)",
    "CREATE TABLE IF NOT EXISTS evo_mem (user_id BIGINT PRIMARY KEY, data TEXT, ts BIGINT)",
]
JUDGE = ("You are a strict grader. Compare CANDIDATE to REFERENCE for correctness and helpfulness.\n"
         "QUESTION: {q}\nREFERENCE: {r}\nCANDIDATE: {c}\nReply with only a number from 0 to 10.")


def install(app, engine, current_user, inference_config):
    def run(sql, **p):
        with engine.begin() as c:
            r = c.execute(text(sql), p)
            return [dict(x._mapping) for x in r] if r.returns_rows else []

    def tok():
        return secrets.token_hex(6)

    def admin(user=Depends(current_user)):
        allowed = {e.strip().lower() for e in os.getenv("AURORA_ADMIN_EMAILS", "").split(",") if e.strip()}
        if (user.get("email") or "").lower() not in allowed:
            raise HTTPException(status_code=403, detail="Admin only. Set AURORA_ADMIN_EMAILS on the server.")
        return user

    @app.on_event("startup")
    def _evo_init():
        try:
            for d in DDL:
                run(d)
        except Exception as exc:
            print("AURORA evolution tables not ready:", exc)

    # ---------- learning signal ----------
    class FbIn(BaseModel):
        prompt: str = Field(max_length=4000)
        answer: str = Field(max_length=8000)
        rating: int = Field(ge=-1, le=1)  # 1 good, -1 bad, 0 = only report uncertainty
        correction: str = Field(default="", max_length=8000)
        uncertain: bool = False

    @app.post("/evo/feedback")
    def feedback(d: FbIn, user=Depends(current_user)):
        now = int(time.time())
        if d.rating != 0:
            run("INSERT INTO evo_feedback VALUES (:i,:u,:p,:a,:r,:c,'pending',:t)", i=tok(), u=user["id"], p=d.prompt,
                a=d.answer, r=d.rating, c=d.correction, t=now)
        if d.rating < 0 or d.uncertain:  # knowledge-gap detection
            key = re.sub(r"\W+", " ", d.prompt.lower()).strip()[:80]
            row = run("SELECT id FROM evo_gaps WHERE gkey=:k", k=key)
            if row:
                run("UPDATE evo_gaps SET n=n+1, ts=:t WHERE id=:i", t=now, i=row[0]["id"])
            else:
                run("INSERT INTO evo_gaps VALUES (:i,:k,:p,1,'open',:t)", i=tok(), k=key, p=d.prompt[:500], t=now)
        return {"ok": True}

    # ---------- admin review ----------
    @app.get("/evo/review")
    def review(user=Depends(admin)):
        return {
            "pending": run("SELECT id,prompt,answer,rating,correction FROM evo_feedback WHERE status='pending' ORDER BY ts DESC LIMIT 30"),
            "gaps": run("SELECT id,prompt,n FROM evo_gaps WHERE status='open' ORDER BY n DESC, ts DESC LIMIT 15"),
            "dataset": run("SELECT COUNT(*) AS n, COALESCE(SUM(holdout),0) AS h FROM evo_data")[0],
            "models": run("SELECT id,name,tag,job,status,score,base_score FROM evo_models ORDER BY ts DESC LIMIT 10"),
            "last_eval": (run("SELECT v FROM evo_meta WHERE k='last_eval'") or [{"v": "null"}])[0]["v"],
        }

    class RevIn(BaseModel):
        action: str
        answer: str = Field(default="", max_length=8000)
        tag: str = Field(default="general", max_length=40)

    @app.post("/evo/review/{fid}")
    def review_item(fid: str, d: RevIn, user=Depends(admin)):
        row = run("SELECT * FROM evo_feedback WHERE id=:i", i=fid)
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        row = row[0]
        if d.action == "reject":
            run("UPDATE evo_feedback SET status='rejected' WHERE id=:i", i=fid)
            return {"ok": True}
        ans = (d.answer or row["correction"] or (row["answer"] if row["rating"] > 0 else "")).strip()
        if not ans:
            raise HTTPException(status_code=400, detail="A bad answer needs a corrected answer before approval.")
        holdout = 1 if int(fid, 16) % 5 == 0 else 0  # ~20% never used for training: kept for honest testing
        run("INSERT INTO evo_data VALUES (:i,:p,:a,:g,:h,:t)", i=tok(), p=row["prompt"], a=ans, g=d.tag, h=holdout, t=int(time.time()))
        run("UPDATE evo_feedback SET status='approved' WHERE id=:i", i=fid)
        key = re.sub(r"\W+", " ", row["prompt"].lower()).strip()[:80]
        run("UPDATE evo_gaps SET status='resolved' WHERE gkey=:k", k=key)
        return {"ok": True, "holdout": bool(holdout)}

    @app.get("/evo/dataset/export", response_class=PlainTextResponse)
    def export(user=Depends(admin)):
        rows = run("SELECT prompt,answer FROM evo_data ORDER BY ts")
        return "\n".join(json.dumps({"messages": [{"role": "user", "content": r["prompt"]},
                                                  {"role": "assistant", "content": r["answer"]}]}) for r in rows)

    # ---------- used by the chat endpoint ----------
    def retrieve(query: str, k: int = 3) -> str:
        """Reviewed answers relevant to the question, injected as reference (learning with no training needed)."""
        qw = set(re.findall(r"\w{4,}", query.lower()))
        if len(qw) < 2:
            return ""
        rows = run("SELECT prompt,answer FROM evo_data ORDER BY ts DESC LIMIT 400")
        scored = sorted(((len(qw & set(re.findall(r"\w{4,}", r["prompt"].lower()))), r) for r in rows), key=lambda x: -x[0])
        best = [r for s, r in scored[:k] if s >= 2]
        if not best:
            return ""
        return ("\n\nReviewed reference answers (verified by the AURORA team; prefer them when relevant, "
                "do not mention them):\n" + "\n---\n".join(f"Q: {r['prompt'][:300]}\nA: {r['answer'][:800]}" for r in best))

    def active() -> list:
        try:
            return [r["name"] for r in run("SELECT name FROM evo_models WHERE status='active'") if r["name"]]
        except Exception:
            return []

    # ---------- evaluation ----------
    async def once(model, content, max_tokens=600):
        base, key, _ = inference_config()
        h = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(base + "/chat/completions", headers=h, json={
                "model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0})
            r.raise_for_status()
            return (r.json()["choices"][0]["message"].get("content") or "").strip()

    async def score(model, items):
        judge = inference_config()[2]
        tot = n = 0
        for it in items:
            try:
                cand = await once(model, it["prompt"])
                g = await once(judge, JUDGE.format(q=it["prompt"][:1500], r=it["answer"][:1500], c=cand[:1500]), 300)
                m = re.search(r"\d+(?:\.\d+)?", g)
                tot += min(10.0, float(m.group(0))) if m else 0
                n += 1
            except Exception:
                continue
        return (tot / n if n else 0.0), n

    async def do_eval(model_id: str = ""):
        items = run("SELECT prompt,answer FROM evo_data WHERE holdout=1 ORDER BY ts DESC LIMIT 20")
        if len(items) < int(os.getenv("AURORA_EVO_MIN_HOLDOUT", "5")):
            raise HTTPException(status_code=400, detail="Not enough held-out reviewed examples yet. Approve more feedback first.")
        base_score, n = await score(inference_config()[2], items)
        out = {"ts": int(time.time()), "n": n, "base_score": round(base_score, 2)}
        if model_id:
            m = run("SELECT * FROM evo_models WHERE id=:i", i=model_id)
            if not m or not m[0]["name"]:
                raise HTTPException(status_code=404, detail="Model not ready")
            cs, _ = await score(m[0]["name"], items)
            run("UPDATE evo_models SET score=:s, base_score=:b WHERE id=:i", s=cs, b=base_score, i=model_id)
            out["candidate_score"] = round(cs, 2)
        run("DELETE FROM evo_meta WHERE k='last_eval'")
        run("INSERT INTO evo_meta VALUES ('last_eval',:v)", v=json.dumps(out))
        return out

    class EvalIn(BaseModel):
        model_id: str = ""

    @app.post("/evo/eval")
    async def eval_now(d: EvalIn, user=Depends(admin)):
        return await do_eval(d.model_id)

    @app.get("/evo/cron")
    async def cron(key: str = Query(min_length=8)):
        """Call from a Render Cron Job / GitHub Action for periodic evaluation (needs AURORA_CRON_KEY)."""
        want = os.getenv("AURORA_CRON_KEY", "")
        if not want or not hmac.compare_digest(want, key):
            raise HTTPException(status_code=403, detail="Bad key")
        return await do_eval("")

    # ---------- fine-tuning (specialists) and gated activation ----------
    def ft_cfg():
        b, k, m = os.getenv("AURORA_FT_BASE_URL", "").rstrip("/"), os.getenv("AURORA_FT_API_KEY", ""), os.getenv("AURORA_FT_BASE_MODEL", "")
        if not (b and k and m):
            raise HTTPException(status_code=501, detail="Set AURORA_FT_BASE_URL, AURORA_FT_API_KEY and AURORA_FT_BASE_MODEL to enable fine-tuning.")
        return b, {"Authorization": f"Bearer {k}"}, m

    class FtIn(BaseModel):
        tag: str = ""

    @app.post("/evo/finetune")
    async def finetune(d: FtIn, user=Depends(admin)):
        b, h, base_model = ft_cfg()
        rows = run("SELECT prompt,answer FROM evo_data WHERE holdout=0" + (" AND tag=:t" if d.tag else ""), t=d.tag)
        if len(rows) < int(os.getenv("AURORA_FT_MIN", "50")):
            raise HTTPException(status_code=400, detail=f"Need at least {os.getenv('AURORA_FT_MIN', '50')} approved training examples (have {len(rows)}).")
        data = "\n".join(json.dumps({"messages": [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["answer"]}]}) for r in rows)
        try:
            async with httpx.AsyncClient(timeout=180) as c:
                up = await c.post(b + "/files", headers=h, files={"file": ("train.jsonl", data.encode())}, data={"purpose": "fine-tune"})
                up.raise_for_status()
                job = await c.post(b + "/fine_tuning/jobs", headers=h, json={"training_file": up.json()["id"], "model": base_model})
                job.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Fine-tune provider error: {str(exc)[:200]}")
        mid = tok()
        run("INSERT INTO evo_models VALUES (:i,'',:g,:j,'training',NULL,NULL,:t)", i=mid, g=d.tag or "general", j=job.json()["id"], t=int(time.time()))
        return {"id": mid, "status": "training"}

    @app.get("/evo/finetune/{mid}")
    async def finetune_status(mid: str, user=Depends(admin)):
        b, h, _ = ft_cfg()
        m = run("SELECT * FROM evo_models WHERE id=:i", i=mid)
        if not m:
            raise HTTPException(status_code=404, detail="Not found")
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.get(b + "/fine_tuning/jobs/" + m[0]["job"], headers=h)
            r.raise_for_status()
            j = r.json()
        if j.get("status") == "succeeded" and j.get("fine_tuned_model"):
            run("UPDATE evo_models SET name=:n, status='candidate' WHERE id=:i", n=j["fine_tuned_model"], i=mid)
        elif j.get("status") in ("failed", "cancelled"):
            run("UPDATE evo_models SET status='failed' WHERE id=:i", i=mid)
        return {"status": j.get("status")}

    @app.post("/evo/models/{mid}/activate")
    def activate(mid: str, user=Depends(admin)):
        m = run("SELECT * FROM evo_models WHERE id=:i", i=mid)
        if not m or m[0]["status"] not in ("candidate", "active"):
            raise HTTPException(status_code=409, detail="Model is not a ready candidate.")
        m = m[0]
        margin = float(os.getenv("AURORA_EVO_MARGIN", "0.3"))
        if m["score"] is None or m["base_score"] is None:
            raise HTTPException(status_code=409, detail="Run an evaluation on this candidate first.")
        if m["score"] < m["base_score"] + margin:
            raise HTTPException(status_code=409, detail=f"Rejected by test: candidate {m['score']:.2f} vs base {m['base_score']:.2f} (needs +{margin}).")
        run("UPDATE evo_models SET status='active' WHERE id=:i", i=mid)
        return {"ok": True}

    @app.post("/evo/models/{mid}/deactivate")
    def deactivate(mid: str, user=Depends(admin)):
        run("UPDATE evo_models SET status='candidate' WHERE id=:i", i=mid)
        return {"ok": True}

    # ---------- cross-session memory sync ----------
    class MemIn(BaseModel):
        data: str = Field(max_length=200_000)

    @app.get("/evo/memory")
    def mem_get(user=Depends(current_user)):
        r = run("SELECT data FROM evo_mem WHERE user_id=:u", u=user["id"])
        return {"data": r[0]["data"] if r else "[]"}

    @app.put("/evo/memory")
    def mem_put(d: MemIn, user=Depends(current_user)):
        run("DELETE FROM evo_mem WHERE user_id=:u", u=user["id"])
        run("INSERT INTO evo_mem VALUES (:u,:d,:t)", u=user["id"], d=d.data, t=int(time.time()))
        return {"ok": True}

    return {"retrieve": retrieve, "active": active}
