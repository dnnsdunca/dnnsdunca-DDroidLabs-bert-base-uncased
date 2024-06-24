class MultimodalAgentModel:
    def __init__(self, text_model_name, image_model_name):
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.image_model = AutoModel.from_pretrained(image_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    def forward(self, text_input, image_input):
        text_embeddings = self.text_model(**self.tokenizer(text_input, return_tensors='pt'))
        image_embeddings = self.image_model(image_input)
        combined_embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
        return combined_embeddings
