"""Tests for snapshots, the write-ahead log, and recovery after a crash."""
import numpy as np
import pytest

from vsearch.cluster import Shard
from vsearch.storage import DurableShard, OutOfOrderWrite, load_snapshot, save_snapshot

PARAMS = dict(M=8, ef_construction=32, ef_search=32, seed=0)
QUERIES = np.random.default_rng(2).standard_normal((10, 16)).astype(np.float32)


def _batches(n_batches=6, per_batch=40, dim=16):
    rng = np.random.default_rng(1)
    vectors = rng.standard_normal((n_batches * per_batch, dim)).astype(np.float32)
    return [(list(range(i, i + per_batch)), vectors[i:i + per_batch])
            for i in range(0, len(vectors), per_batch)]


def _results(shard):
    return [shard.search(q, 5) for q in QUERIES]


def _reference(batches):
    shard = Shard(**PARAMS)
    for ids, vectors in batches:
        shard.add_batch(ids, vectors)
    return shard


def test_snapshot_round_trip_keeps_building_the_same_graph(tmp_path):
    batches = _batches()
    original = _reference(batches[:3])
    save_snapshot(original, last_seq=3, path=tmp_path / "snapshot.npz")

    restored, last_seq = load_snapshot(tmp_path / "snapshot.npz")
    assert last_seq == 3
    assert _results(restored) == _results(original)

    # Keep writing to both: the restored random generator must make the same level choices.
    for ids, vectors in batches[3:]:
        original.add_batch(ids, vectors)
        restored.add_batch(ids, vectors)
    assert restored.index.graph == original.index.graph


def test_without_the_saved_generator_state_the_graph_diverges(tmp_path):
    # What a restart would do if the snapshot didn't store the generator: reseed it.
    batches = _batches()
    original = _reference(batches[:3])
    save_snapshot(original, last_seq=3, path=tmp_path / "snapshot.npz")
    restored, _ = load_snapshot(tmp_path / "snapshot.npz")
    restored.index.rng = np.random.default_rng(PARAMS["seed"])

    for ids, vectors in batches[3:]:
        original.add_batch(ids, vectors)
        restored.add_batch(ids, vectors)
    assert restored.index.graph != original.index.graph


def test_restart_replays_the_log(tmp_path):
    batches = _batches()
    shard = DurableShard(tmp_path, snapshot_every=0, **PARAMS)    # never snapshot: all in the log
    for ids, vectors in batches:
        shard.add_batch(ids, vectors)
    shard.close()

    recovered = DurableShard(tmp_path, snapshot_every=0, **PARAMS)
    assert recovered.replayed_records == len(batches)
    assert _results(recovered) == _results(_reference(batches))


def test_restart_from_snapshot_plus_log_tail(tmp_path):
    batches = _batches()                                           # 6 batches x 40 vectors
    shard = DurableShard(tmp_path, snapshot_every=150, **PARAMS)   # snapshot after batch 4
    for ids, vectors in batches:
        shard.add_batch(ids, vectors)
    shard.close()

    recovered = DurableShard(tmp_path, snapshot_every=150, **PARAMS)
    assert recovered.snapshot_seq == 4
    assert recovered.replayed_records == 2                         # batches 5 and 6
    assert _results(recovered) == _results(_reference(batches))


def test_torn_last_record_is_cut_off(tmp_path):
    batches = _batches()
    wal = tmp_path / "wal.log"
    shard = DurableShard(tmp_path, snapshot_every=0, **PARAMS)
    for ids, vectors in batches[:4]:
        shard.add_batch(ids, vectors)
    shard.close()
    intact_size = wal.stat().st_size

    # A crash in the middle of appending record 5 leaves only part of it on disk.
    shard = DurableShard(tmp_path, snapshot_every=0, **PARAMS)
    shard.add_batch(*batches[4])
    shard.close()
    with open(wal, "r+b") as f:
        f.truncate(intact_size + (wal.stat().st_size - intact_size) // 2)

    recovered = DurableShard(tmp_path, snapshot_every=0, **PARAMS)
    assert recovered.replayed_records == 4
    assert wal.stat().st_size == intact_size                       # the torn bytes are gone

    # The log keeps working after the repair.
    recovered.add_batch(*batches[4])
    recovered.close()
    again = DurableShard(tmp_path, snapshot_every=0, **PARAMS)
    assert again.replayed_records == 5
    assert _results(again) == _results(_reference(batches[:5]))


def test_corrupted_record_ends_the_log(tmp_path):
    batches = _batches()
    shard = DurableShard(tmp_path, snapshot_every=0, **PARAMS)
    for ids, vectors in batches[:3]:
        shard.add_batch(ids, vectors)
    shard.close()

    wal = tmp_path / "wal.log"
    data = bytearray(wal.read_bytes())
    data[-1] ^= 0xFF                                               # flip bits in the last record
    wal.write_bytes(bytes(data))

    recovered = DurableShard(tmp_path, snapshot_every=0, **PARAMS)
    assert recovered.replayed_records == 2                         # its checksum no longer matches


def test_crash_between_snapshot_and_log_reset_does_not_duplicate(tmp_path):
    batches = _batches()
    wal = tmp_path / "wal.log"
    shard = DurableShard(tmp_path, snapshot_every=0, **PARAMS)
    for ids, vectors in batches:
        shard.add_batch(ids, vectors)
    old_log = wal.read_bytes()
    shard.checkpoint()                                             # snapshot written, log emptied
    shard.close()
    wal.write_bytes(old_log)                                       # as if the crash hit before the reset

    recovered = DurableShard(tmp_path, snapshot_every=0, **PARAMS)
    assert recovered.replayed_records == 0                         # every record is already in the snapshot
    assert len(recovered) == sum(len(ids) for ids, _ in batches)
    assert _results(recovered) == _results(_reference(batches))


def test_out_of_order_write_is_rejected(tmp_path):
    batches = _batches()
    shard = DurableShard(tmp_path, snapshot_every=0, **PARAMS)
    shard.add_batch(*batches[0], seq=1)
    with pytest.raises(OutOfOrderWrite):
        shard.add_batch(*batches[1], seq=3)                        # seq 2 is missing
    with pytest.raises(OutOfOrderWrite):
        shard.add_batch(*batches[1], seq=1)                        # seq 1 is already applied
    assert shard.last_seq == 1
    assert len(shard) == len(batches[0][0])


def test_records_after_says_when_the_log_no_longer_covers_them(tmp_path):
    batches = _batches()
    shard = DurableShard(tmp_path, snapshot_every=0, **PARAMS)
    for ids, vectors in batches[:3]:
        shard.add_batch(ids, vectors)
    assert [record[0] for record in shard.records_after(1)] == [2, 3]

    shard.checkpoint()                                             # seqs 1-3 now only in the snapshot
    shard.add_batch(*batches[3])
    assert shard.records_after(1) is None
    assert [record[0] for record in shard.records_after(3)] == [4]


def test_install_snapshot_copies_a_peer_exactly(tmp_path):
    batches = _batches()
    peer = DurableShard(tmp_path / "peer", snapshot_every=0, **PARAMS)
    for ids, vectors in batches:
        peer.add_batch(ids, vectors)
    data, last_seq = peer.snapshot_bytes()

    copy = DurableShard(tmp_path / "copy", snapshot_every=0, **PARAMS)
    copy.install_snapshot(data)
    assert copy.last_seq == last_seq == len(batches)
    assert _results(copy) == _results(peer)

    copy.close()                                                   # and it survives a restart
    assert _results(DurableShard(tmp_path / "copy", snapshot_every=0, **PARAMS)) == _results(peer)
