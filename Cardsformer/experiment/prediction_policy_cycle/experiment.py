from experiment.prediction_policy_cycle.train_policy_model import execute

if __name__ == "__main__":
    iter = 1
    execute(
        train_data_limit = 1000,
        data_num = 10,
        model_save_dir = f"experiment/prediction_policy_cycle/policy_models/iter_{iter}"
    )
    
    print("prediction process start")

