from utils import *
import eval




def main():
    prompts = ["The movie was great, I loved it.","The movie was terrible, I hated it.", "I love dogs"]
    model_name = "bigscience/bloom-560m"
    model = AutoModel.from_pretrained(model_name, output_attentions=True)#.to(device)  # Configure model to return attention values
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_path = duplicate_prune_model(prompts, "bloom-560m", model, tokenizer, metric='euclidean', verbose=True)
    # model_path = "models/bloom-560m_euclidean_0.5"
    eval.evaluate(
        model_path= model_path,
        model_name="bloom-560m",
        task="lambada_openai,paws_en"
    )

if __name__ == "__main__":
    main()