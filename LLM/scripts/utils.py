PATH = "/content/state.pth"

torch.save(model.state_dict(),PATH)

from google.colab import drive
drive.mount('/content/gdrive')

