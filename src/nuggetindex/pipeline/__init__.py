"""Pipeline: canonicalize -> temporal inference -> dedup -> conflict resolution.

The four-stage pipeline turns raw extractor output into store-ready, governed
nuggets. See spec §5.3-§5.6 for details. Orchestrated by
``DocumentConstructor`` and wired into ``NuggetStore.aingest``.
"""
