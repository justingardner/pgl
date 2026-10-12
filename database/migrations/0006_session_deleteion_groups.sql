-- Record coordinated session deletion groups.
--
-- This migration adds storage only. Existing deletion markers are unchanged.
-- Historical groups are not inferred from existing timestamps.
--
-- Lifecycle code will coordinate:
--   * group creation and membership,
--   * session/file/content/location deletion markers,
--   * whole-group restoration,
--   * parent-visibility requirements.
--
-- These tables alone do not enforce the complete lifecycle protocol.

CREATE TABLE session_deletion_groups (
    deletion_group_id bigint GENERATED ALWAYS AS IDENTITY
        (START WITH 0 MINVALUE 0) PRIMARY KEY,
    top_session_id bigint NOT NULL REFERENCES session(session_id),
    created_at timestamptz NOT NULL DEFAULT statement_timestamp(),
    restored_at timestamptz
);

-- A session may lead multiple historical groups, but only one
-- outstanding group at a time.
CREATE UNIQUE INDEX session_deletion_groups_active_top_unique
ON session_deletion_groups (top_session_id)
WHERE restored_at IS NULL;

CREATE TABLE session_deletion_group_members (
    deletion_group_id bigint NOT NULL
        REFERENCES session_deletion_groups(deletion_group_id),
    session_id bigint NOT NULL REFERENCES session(session_id),
    restored_at timestamptz,
    PRIMARY KEY (deletion_group_id, session_id)
);

-- A session may belong to multiple historical groups, but only one
-- outstanding deletion group at a time.
CREATE UNIQUE INDEX session_deletion_group_members_active_session_unique
ON session_deletion_group_members (session_id)
WHERE restored_at IS NULL;

-- Support historical membership lookup by session.
CREATE INDEX session_deletion_group_members_session_index
ON session_deletion_group_members (session_id);

-- On restoration, lifecycle code will set the group's restored_at and
-- every member's restored_at to the same timestamp, atomically.
--
-- Membership rows are retained. A subsequent deletion creates a new group.
-- Previously deleted descendants are not enrolled in the new group.