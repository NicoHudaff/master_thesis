CREATE TABLE master_thesis.results_training (
	lr FLOAT,
	opt VARCHAR,
	n_layers INT,
    outs VARCHAR,
    drops VARCHAR,
    epochs INT,
    minutes INT,
    strict BOOLEAN,
    train_loss FLOAT,
    epoch INT,
    test_loss FLOAT,
    anomaly_loss FLOAT,
    f1_test FLOAT,
    valid_loss FLOAT,
    valid_f1 FLOAT
);