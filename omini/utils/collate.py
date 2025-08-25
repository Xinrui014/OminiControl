# somewhere accessible, e.g., omini/data/collate.py
def simple_collate(batch):
    assert len(batch) == 1, "Use batch_size=1 or implement padding for variable-K boxes."
    return batch[0]
