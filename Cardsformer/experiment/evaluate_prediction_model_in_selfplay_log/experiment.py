from experiment.evaluate_prediction_model_in_selfplay_log.evaluate_prediction_model import evaluate_prediction_model
from experiment.evaluate_prediction_model_in_selfplay_log.generate_play_data import generate_selfplay_log

import csv





if __name__ == "__main__":
    # choose lastest prediction model
    prediction_model_tar_path = "./trained_models/prediction_model4715.tar" 

    # choose lastest policy model
    policy_model_checkpoint_path = "../../cf_policy_1gpu/Cardsformer/trained_policy_model/Cardsformer/Trained_weights_66000000.ckpt"

    # generate_selfplay_log
    # generate_selfplay_log(prediction_model_tar_path, policy_model_checkpoint_path)

    # evaluate_model
    candidate_eval_data_path = [
        ["./off_line_data_vs_policy_model.npy"],
        ["./off_line_data9.npy"],
        ["./off_line_data_vs_ai0.npy"]
    ]
    
    candidate_models = [
        "../../enhance-cardsformer-batch-almost-5000/Cardsformer/trained_models/prediction_model10.tar",
        "../../enhance-cardsformer-batch-almost-5000/Cardsformer/trained_models/prediction_model100.tar",
        "../../enhance-cardsformer-batch-almost-5000/Cardsformer/trained_models/prediction_model201.tar",
        "../../enhance-cardsformer-batch-almost-5000/Cardsformer/trained_models/prediction_model300.tar",
        "../../enhance-cardsformer-batch-almost-5000/Cardsformer/trained_models/prediction_model408.tar",
        "../../enhance-cardsformer-batch-almost-5000/Cardsformer/trained_models/prediction_model500.tar",
        "../../enhance-cardsformer-batch-almost-5000/Cardsformer/trained_models/prediction_model622.tar",
        "../../enhance-cardsformer-batch-almost-5000/Cardsformer/trained_models/prediction_model739.tar",
        "../../enhance-cardsformer-batch-almost-5000/Cardsformer/trained_models/prediction_model1182.tar",
        "../../enhance-cardsformer-batch-almost-5000/Cardsformer/trained_models/prediction_model4715.tar"
    ]

    with open("./experiment/evaluate_prediction_model_in_selfplay_log/output.csv", mode="w", newline="") as file:
        writer = csv.writer(file)

        # header
        writer.writerow(["model_name", "eval_data", "loss"])


        for eval_data_list in candidate_eval_data_path:
            for prediction_model_tar_path in candidate_models:
                res = evaluate_prediction_model(prediction_model_tar_path, eval_data_list)

                writer.writerow([prediction_model_tar_path, eval_data_list[0], res])

