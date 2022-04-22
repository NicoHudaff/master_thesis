CREATE MATERIALIZED VIEW master_thesis.materialized_view_raw_data_match AS (
    SELECT
        DISTINCT match_id
    FROM
        master_thesis.raw_data
);