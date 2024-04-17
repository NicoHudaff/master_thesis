CREATE TABLE master_thesis.raw_data (
	id VARCHAR UNIQUE NOT NULL,
	type VARCHAR,
	match_id INT,
	minute INT,
	period FLOAT,
	second INT,
	team VARCHAR,
	result VARCHAR,
	content VARCHAR
);