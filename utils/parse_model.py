import models

def initialize_model(config):
    # Check if the model exists in classification.models
    model_class = config.model
    model = getattr(models, model_class)
    model = model()

    return model