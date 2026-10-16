"""Agent memory on a real cluster: facts come back by meaning, replaced and forgotten ones
do not, newer facts win ties, and owners cannot see each other's memories."""
import json

import pytest

from vsearch.cli import main
from vsearch.health import cluster_health
from vsearch.mcp_server import build_memory_server
from vsearch.memory import HALF_LIFE_DAYS, RECENCY_WEIGHT, MemoryStore, describe_age
from vsearch.state import UserError, read_state

DAY = 86400


def run(home, *argv):
    return main(["--home", str(home), *argv])


class Clock:
    """A clock the test moves by hand."""
    def __init__(self):
        self.now = 1_800_000_000.0

    def __call__(self):
        return self.now


@pytest.fixture
def home(tmp_path, capsys):
    home = tmp_path / "memory"
    assert run(home, "cluster", "up", "--shards", "2", "--replicas", "2") == 0
    capsys.readouterr()
    yield home
    run(home, "cluster", "down", "--wipe")


@pytest.fixture
def store(home):
    clock = Clock()
    store = MemoryStore(home, "rahul", embedder="hash", clock=clock)
    store.clock = clock
    yield store
    store.close()


def texts(memories):
    return [memory.text for memory in memories]


def test_facts_come_back_by_what_they_are_about(store, home):
    store.remember("Rahul prefers short answers with examples.", tags=["preference"])
    store.remember("The project uses gRPC between the coordinator and the shards.")
    store.remember("Rahul is applying for backend and AI engineer roles.", tags=["career"])
    found = store.recall("how long should answers be", k=2)
    assert found[0].text == "Rahul prefers short answers with examples."
    assert found[0].tags == ["preference"] and 0 < found[0].similarity <= 1
    assert texts(store.recall("what roles is he applying for", k=1)) == \
        ["Rahul is applying for backend and AI engineer roles."]
    assert texts(store.recall("anything", tags=["career"])) == \
        ["Rahul is applying for backend and AI engineer roles."]
    # The cluster now records that it holds memories, made with the hash embedder.
    assert read_state(home)["dataset"] == {"name": "memory", "embedder": "hash", "dim": 512}


def test_the_same_fact_is_not_stored_twice(store):
    first, stored = store.remember("Rahul prefers short answers.")
    again, stored_again = store.remember("  Rahul prefers short answers.  ")
    assert stored and not stored_again and again == first
    assert len(store.recall("short answers", k=10)) == 1


def test_a_replaced_fact_is_no_longer_recalled(store):
    old, _ = store.remember("Rahul lives in Delhi.")
    store.clock.now += 10 * DAY
    new, _ = store.remember("Rahul moved to Bangalore.", replaces=[old])
    assert texts(store.recall("where does Rahul live", k=5)) == ["Rahul moved to Bangalore."]
    with pytest.raises(UserError, match="no current memory"):
        store.remember("Rahul lives in Pune.", replaces=[old])      # already replaced
    with pytest.raises(UserError, match="no current memory"):
        store.remember("Something.", replaces=["m-not-an-id"])
    # The old fact may be stored again later: it is a new memory then.
    again, stored = store.remember("Rahul lives in Delhi.")
    assert stored and again != old


def test_a_forgotten_fact_is_no_longer_recalled(store):
    keep, _ = store.remember("Rahul likes tea.")
    gone, _ = store.remember("Rahul's phone number is in his notes.")
    store.forget(gone)
    assert texts(store.recall("Rahul", k=5)) == ["Rahul likes tea."]
    for bad in (gone, "m-unknown"):
        with pytest.raises(UserError, match="no current memory"):
            store.forget(bad)


def test_newer_facts_win_when_they_match_equally(store):
    store.remember("The favourite colour is blue.")
    store.clock.now += 60 * DAY
    store.remember("The favourite colour is green.")
    best, other = store.recall("favourite colour", k=2)
    assert best.text == "The favourite colour is green."
    # Both match about equally; the gap in score is the recency bonus.
    assert abs(best.similarity - other.similarity) < 0.05
    bonus = lambda days: RECENCY_WEIGHT * 0.5 ** (days / HALF_LIFE_DAYS)
    assert best.score - best.similarity == pytest.approx(bonus(0))
    assert other.score - other.similarity == pytest.approx(bonus(60))


def test_owners_cannot_see_each_others_memories(store, home):
    other = MemoryStore(home, "priya", embedder="hash")
    try:
        other.remember("Priya prefers long, detailed answers.")
        store.remember("Rahul prefers short answers.")
        assert texts(other.recall("answers", k=5)) == ["Priya prefers long, detailed answers."]
        assert texts(store.recall("answers", k=5)) == ["Rahul prefers short answers."]
        with pytest.raises(UserError):
            other.forget(store.recall("answers")[0].id)          # not hers to forget
    finally:
        other.close()
    # Each owner is a namespace, so each one's memories sit on a single shard (twice: replicas).
    sizes = [server.size for server in cluster_health(read_state(home))]
    assert sorted(sizes) == [1, 1, 1, 1]


def test_memories_survive_a_restart(store, home, capsys):
    memory_id, _ = store.remember("Rahul's interview is on Friday.")
    store.close()
    assert run(home, "cluster", "down") == 0 and run(home, "cluster", "up") == 0
    capsys.readouterr()
    found = store.recall("when is the interview")
    assert [(m.id, m.text) for m in found] == [(memory_id, "Rahul's interview is on Friday.")]


def test_a_documents_cluster_is_not_used_for_memory(home, tmp_path, capsys):
    (tmp_path / "a.md").write_text("Some text.\n")
    assert run(home, "ingest", str(tmp_path / "a.md"), "--embedder", "hash") == 0
    store = MemoryStore(home, "rahul", embedder="hash")
    with pytest.raises(UserError, match="cluster of their own"):
        store.remember("A fact.")
    with pytest.raises(UserError, match="cluster of their own"):
        store.recall("anything")
    with pytest.raises(UserError, match="nothing to remember"):
        MemoryStore(home, "rahul").remember("   ")
    with pytest.raises(UserError, match="owner"):
        MemoryStore(home, "")


def test_describe_age():
    assert [describe_age(s) for s in (5, 60, 7199, 3 * DAY)] == \
        ["just now", "1 minute ago", "1 hour ago", "3 days ago"]


def test_memory_on_the_command_line(home, capsys):
    assert run(home, "memory", "remember", "Rahul prefers short answers.", "--owner", "rahul",
               "--tag", "preference", "--embedder", "hash") == 0
    memory_id = capsys.readouterr().out.split()[-1]
    assert run(home, "memory", "remember", "Rahul prefers short answers.", "--owner", "rahul") == 0
    assert capsys.readouterr().out.strip() == f"already remembered as {memory_id}"
    assert run(home, "memory", "recall", "how should answers look", "--owner", "rahul") == 0
    output = capsys.readouterr().out
    assert memory_id in output and "[preference] Rahul prefers short answers." in output
    assert run(home, "memory", "recall", "anything", "--owner", "priya") == 0
    assert "nothing remembered matches" in capsys.readouterr().out
    assert run(home, "memory", "forget", memory_id, "--owner", "rahul") == 0
    assert "its text stays on disk" in capsys.readouterr().out
    assert run(home, "memory", "forget", memory_id, "--owner", "rahul") == 1
    assert "no current memory" in capsys.readouterr().err


def test_memory_over_mcp(home):
    server = build_memory_server(home, "rahul", embedder="hash")

    def call(name, **arguments):
        reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                               "params": {"name": name, "arguments": arguments}})
        return reply["result"]["content"][0]["text"], reply["result"]["isError"]

    try:
        started = server.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
        assert "long-term memory of rahul" in started["result"]["instructions"]
        listed = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
        assert [tool["name"] for tool in listed] == ["recall", "remember", "forget"]

        assert call("recall", query="anything") == ("Nothing remembered matches.", False)
        text, _ = call("remember", text="Rahul lives in Delhi.", tags=["home"])
        old = text.split()[-1].rstrip(".")
        text, _ = call("remember", text="Rahul moved to Bangalore.", replaces=[old], tags=["home"])
        new = text.split()[-1].rstrip(".")
        text, is_error = call("recall", query="where does Rahul live")
        assert not is_error and f"[{new}] (just now; tags: home) Rahul moved to Bangalore." in text
        assert old not in text
        assert call("forget", memory_id=new) == (f"Forgot {new}. It will no longer be recalled.", False)
        text, is_error = call("forget", memory_id=new)
        assert is_error and "no current memory" in text
        text, is_error = call("remember", text="")
        assert is_error and "nothing to remember" in text
    finally:
        server.close()
