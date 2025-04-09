docker run -it --gpus all --name cardsformer_env_container -v "$(pwd)/..:/app" namelessogya/cardsformer_env bash
docker exec -it cardsformer_env_container bash