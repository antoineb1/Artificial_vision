import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class MultiTaskMobileNetV2(nn.Module):
    """
    Multi-task MobileNetV2 model for gender, hat, bag, top color, and bottom color classification.

    Args:
        num_output_gender (int): Number of output classes for gender classification.
        num_output_hat (int): Number of output classes for hat classification.
        num_output_bag (int): Number of output classes for bag classification.
        num_output_top_color (int): Number of output classes for top color classification.
        num_output_bottom_color (int): Number of output classes for bottom color classification.
    """
    def __init__(self, num_output_gender, num_output_hat, num_output_bag, num_output_top_color, num_output_bottom_color):
        super(MultiTaskMobileNetV2, self).__init__()

        # Load the pre-trained MobileNetV2 base
        self.base_model = models.mobilenet_v2()

        # Freeze all layers up to the second-to-last layer
        for name, param in self.base_model.named_parameters():
            if not name.startswith('classifier') and not name.startswith('features.18'):  # Ignore the last fully connected layer
                param.requires_grad = False

        # Modify the last fully connected layer to adapt to the tasks
        self.base_model.classifier[-1] = nn.Identity()

        # Add specific branches for each task
        self.gender_fc = nn.Linear(1280, num_output_gender)
        self.hat_fc = nn.Linear(1280, num_output_hat)
        self.bag_fc = nn.Linear(1280, num_output_bag)
        self.top_color_fc = nn.Linear(1280, num_output_top_color)
        self.bottom_color_fc = nn.Linear(1280, num_output_bottom_color)

    def forward(self, x):
        """
        Forward pass through the shared MobileNetV2 base and task-specific branches.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Output tensors for gender, hat, bag, top color, and bottom color tasks.
        """
        shared_features = self.base_model(x)

        output_gender = self.gender_fc(shared_features)
        output_hat = self.hat_fc(shared_features)
        output_bag = self.bag_fc(shared_features)
        output_top_color = self.top_color_fc(shared_features)
        output_bottom_color = self.bottom_color_fc(shared_features)

        return output_gender, output_hat, output_bag, output_top_color, output_bottom_color


class MultiTaskPAR:
    """
    Multi-task wrapper for the MultiTaskMobileNetV2 model.

    Args:
        weights_path (str): Path to the pre-trained model weights.
        num_output_gender (int): Number of output classes for gender classification.
        num_output_hat (int): Number of output classes for hat classification.
        num_output_bag (int): Number of output classes for bag classification.
        num_output_top_color (int): Number of output classes for top color classification.
        num_output_bottom_color (int): Number of output classes for bottom color classification.
    """
    def __init__(self, weights_path, num_output_gender=1, num_output_hat=1, num_output_bag=1, num_output_top_color=11, num_output_bottom_color=11):
        # Initialize the MultiTaskMobileNetV2 model
        self.model = MultiTaskMobileNetV2(num_output_gender, num_output_hat, num_output_bag, num_output_top_color, num_output_bottom_color)

        # Define activation functions
        self.binary_activation = nn.Sigmoid()  
        self.multiclass_activation = nn.Softmax(dim=1)  

        # Image preprocessing pipeline
        self.image_preprocessing = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load pre-trained weights
        self.model.load_state_dict(torch.load(weights_path))

    def to(self, mode):
        if mode == "cuda":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(device)
        else:
            self.model = self.model.to("cpu")

    def extract_attributes(self, image):
        """
        Extract attributes (gender, hat, bag, top color, bottom color) from the input image.

        Args:
            image (PIL.Image): Input image.

        Returns:
            tuple: Extracted attributes - gender, hat, bag, top color, bottom color.
        """
        input_tensor = self.image_preprocessing(image).unsqueeze(0).to("cuda")

        with torch.no_grad():
            outputs = self.model(input_tensor)

        gender_label = "female" if self.binary_activation(outputs[0]).item() >= 0.5 else "male"
        hat_label = True if self.binary_activation(outputs[1]).item() >= 0.5 else False
        bag_label = True if self.binary_activation(outputs[2]).item() >= 0.5 else False

        color_mapping = {
            1: "black",
            2: "blue",
            3: "brown",
            4: "gray",
            5: "green",
            6: "orange",
            7: "pink",
            8: "purple",
            9: "red",
            10: "white",
            11: "yellow"
        }

        top_color_label = color_mapping[torch.argmax(self.multiclass_activation(outputs[3])).item() + 1]
        bottom_color_label = color_mapping[torch.argmax(self.multiclass_activation(outputs[4])).item() + 1]

        return gender_label, hat_label, bag_label, top_color_label, bottom_color_label
