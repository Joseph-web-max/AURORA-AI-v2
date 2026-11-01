"""Server-side storage for chats and projects (the browser becomes a cache).

Per-item sync: every chat/project is its own row, newest edit wins per item, deletes propagate,
and a one-time migration imports the old single-blob /chats data.
"""
import json
import secrets
import time

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

DDL = ("CREATE TABLE IF NOT EXISTS sync_items (user_id BIGINT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL, "
       "data TEXT NOT NULL, updated_at BIGINT NOT NULL, deleted INT NOT NULL, sts BIGINT NOT NULL, PRIMARY KEY(user_id, id))")
PAGE = 200


def install(app, engine, current_user):
    def run(sql, **p):
        with engine.begin() as c:
            r = c.execute(text(sql), p)
            return [dict(x._mapping) for x in r] if r.returns_rows else []

    @app.on_event("startup")
    def _sync_init():
        try:
            run(DDL)
            run("CREATE TABLE IF NOT EXISTS shares (id TEXT PRIMARY KEY, user_id BIGINT NOT NULL, title TEXT, data TEXT NOT NULL, ts BIGINT NOT NULL)")
        except Exception as exc:
            print("AURORA sync table not ready:", exc)

    def migrate_legacy(uid: int, now: int) -> None:
        """One time: copy the old single-blob chat backup into per-chat rows."""
        try:
            row = run("SELECT data, updated_at FROM user_chats WHERE user_id=:u", u=uid)
            if not row:
                return
            with engine.begin() as c:
                for ch in json.loads(row[0]["data"] or "[]"):
                    cid = str(ch.get("id") or ch.get("created") or ch.get("title") or "")[:64]
                    if not cid or len(json.dumps(ch)) > 900_000:
                        continue
                    ua = int(ch.get("updated") or ch.get("created") or row[0]["updated_at"] or 0)
                    c.execute(text("INSERT INTO sync_items VALUES (:u,:i,'chat',:d,:a,0,:s)"),
                              {"u": uid, "i": cid, "d": json.dumps({**ch, "id": cid}), "a": ua, "s": now})
        except Exception as exc:
            print("AURORA legacy chat migration skipped:", exc)

    @app.get("/sync")
    def pull(since: int = Query(default=0, ge=0), user=Depends(current_user)):
        now = int(time.time() * 1000)
        if since == 0 and not run("SELECT 1 AS x FROM sync_items WHERE user_id=:u LIMIT 1", u=user["id"]):
            migrate_legacy(user["id"], now)
        rows = run("SELECT id, kind, data, updated_at, deleted, sts FROM sync_items WHERE user_id=:u AND sts>=:s "
                   "ORDER BY sts, id LIMIT :n", u=user["id"], s=since, n=PAGE + 1)
        more = len(rows) > PAGE
        rows = rows[:PAGE]
        return {"items": rows, "more": more, "cursor": rows[-1]["sts"] if more else now}

    class Item(BaseModel):
        id: str = Field(min_length=1, max_length=64)
        kind: str
        data: str = Field(max_length=900_000)
        updated_at: int
        deleted: int = 0

    class PushIn(BaseModel):
        items: list[Item]

    @app.post("/sync")
    def push(body: PushIn, user=Depends(current_user)):
        if len(body.items) > 50:
            raise HTTPException(status_code=413, detail="Too many items in one request")
        now = int(time.time() * 1000)
        applied = 0
        with engine.begin() as c:
            for it in body.items:
                if it.kind not in ("chat", "project"):
                    continue
                ua = min(it.updated_at, now + 300_000)  # ignore clocks far in the future
                cur = c.execute(text("SELECT updated_at FROM sync_items WHERE user_id=:u AND id=:i"),
                                {"u": user["id"], "i": it.id}).first()
                if cur and cur[0] >= ua:
                    continue  # server already has a newer version
                c.execute(text("DELETE FROM sync_items WHERE user_id=:u AND id=:i"), {"u": user["id"], "i": it.id})
                c.execute(text("INSERT INTO sync_items VALUES (:u,:i,:k,:d,:a,:x,:s)"),
                          {"u": user["id"], "i": it.id, "k": it.kind, "d": it.data, "a": ua,
                           "x": 1 if it.deleted else 0, "s": now})
                applied += 1
        return {"applied": applied, "now": now}

    class ShareIn(BaseModel):
        title: str = Field(default="Chat", max_length=200)
        data: str = Field(max_length=900_000)  # JSON list of {role, content}

    @app.post("/share")
    def share_create(body: ShareIn, user=Depends(current_user)):
        sid = secrets.token_urlsafe(9)
        run("INSERT INTO shares VALUES (:i,:u,:t,:d,:s)", i=sid, u=user["id"], t=body.title, d=body.data, s=int(time.time()))
        return {"id": sid}

    @app.get("/share/{sid}")
    def share_get(sid: str):
        """Public read-only: anyone with the link can read."""
        r = run("SELECT title, data FROM shares WHERE id=:i", i=sid[:32])
        if not r:
            raise HTTPException(status_code=404, detail="This shared chat does not exist or was removed.")
        return r[0]

    @app.delete("/share/{sid}")
    def share_delete(sid: str, user=Depends(current_user)):
        run("DELETE FROM shares WHERE id=:i AND user_id=:u", i=sid[:32], u=user["id"])
        return {"ok": True}
