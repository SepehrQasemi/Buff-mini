# Stage-28 Test Plan

## Windowing (Stage-28.1)
- `tests/test_stage28_window_calendar_counts.py`
  - Verifies exact rolling window counts on a deterministic 48-month synthetic timeline.
  - Asserts 3m/1m produces 46 windows and 6m/1m produces 42 windows.
- `tests/test_stage28_window_calendar_no_overlap_bug.py`
  - Verifies monotonic window starts/ends and exact month-step progression.
  - Guards against off-by-one and malformed calendar boundaries.

## Notes
- All tests are offline and deterministic.
- Window generation uses UTC timestamps and deterministic month offsets.
