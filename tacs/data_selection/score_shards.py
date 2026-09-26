"""Contiguous score shards for one candidate loss pass (no torch import; shared by the scorer and tests).

A shard scores the 1-based batches start < i <= end of an unbucketed candidate loader. Boundaries are
multiples of the loss-chunk save interval (the last is the batch count), so each shard writes exactly the
losses-<batch>.pt files an unsharded pass writes and every batch keeps its composition: the shards' caches
together are the unsharded cache.
"""


def shard_batch_range(total_batches, shard, num_shards, save_interval):
    """(start, end] batch range of shard `shard` (0-based) of `num_shards`, or None when the shard is empty."""
    total_batches, shard, num_shards = int(total_batches), int(shard), int(num_shards)
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError("invalid score shard %d/%d" % (shard, num_shards))
    interval = max(1, int(save_interval))

    def boundary(j):
        if j >= num_shards:
            return total_batches
        return min(total_batches, -(-(total_batches * j) // (num_shards * interval)) * interval)

    start, end = boundary(shard), boundary(shard + 1)
    return (start, end) if end > start else None
