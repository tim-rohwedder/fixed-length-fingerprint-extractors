import torch
import torchvision


class LocalizationNetwork(torch.torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The localization net uses a downsampled version of the image for performance
        self.input_size = (128, 128)
        self.resize = torchvision.transforms.Resize(
            size=self.input_size, antialias=True
        )
        # Spatial transformer localization-network
        self.localization = torch.nn.Sequential(
            torch.nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2, stride=2),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = torch.nn.Sequential(
            torch.nn.Linear(8 * 8 * 64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 3)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resized_x = self.resize(x)
        xs = self.localization(resized_x)
        xs = xs.view(-1, 8 * 8 * 64)
        theta_x_y = self.fc_loc(xs)
        theta_x_y = theta_x_y.view(-1, 3)
        theta = theta_x_y[:, 0]  # Rotation angle
        # Construct rotation and scaling matrix
        m11 = torch.cos(theta)
        m12 = -torch.sin(theta)
        m13 = theta_x_y[:, 1]  # offset x
        m21 = torch.sin(theta)
        m22 = torch.cos(theta)
        m23 = theta_x_y[:, 2]  # offset y

        mat = torch.concatenate((m11, m12, m13, m21, m22, m23))
        mat = mat.view(-1, 2, 3)
        grid = torch.nn.functional.affine_grid(mat, x.size(), align_corners=False)
        x = torch.nn.functional.grid_sample(x, grid, align_corners=False)
        return x


def main():
    model = LocalizationNetwork()
    print("no syntax errors")


if __name__ == "__main__":
    main()
