import torch
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import open3d as o3d
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image

# Configuration
IMAGE_PATH = "./sci_fi_gun.jpg"  # Update this path
DEPTH_STRENGTH = 3.0             # 1-5 (Higher = more depth)
OUTPUT_SCALE = 0.03              # Overall size (0.01-0.1)

def load_midas_model(device):
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
    model.to(device).eval()
    transform = Compose([Resize((384, 384)), ToTensor()])
    return model, transform

def estimate_depth(image_path, model, transform, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        depth = model(input_tensor).squeeze().cpu().numpy()
    
    # Proper depth processing
    depth = cv2.medianBlur(depth, 5)
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    return 1 - depth  # Invert depth for correct orientation

def create_3d_model(depth, image_path):
    # Process image
    img = Image.open(image_path).convert("RGB")
    img = img.resize(depth.shape[::-1])
    colors = np.array(img).reshape(-1, 3) / 255.0
    
    # Create point cloud
    h, w = depth.shape
    y, x = np.mgrid[0:h, 0:w]
    points = np.stack([
        x.flatten() * OUTPUT_SCALE,
        y.flatten() * OUTPUT_SCALE,
        depth.flatten() * DEPTH_STRENGTH
    ], axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Robust mesh generation
    pcd.estimate_normals()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([0.005, 0.01, 0.02]))
    
    # Cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    
    output_path = f"gun_model_{int(time.time())}.obj"
    o3d.io.write_triangle_mesh(output_path, mesh)
    return mesh, output_path

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_midas_model(device)
    
    # Get depth map
    depth = estimate_depth(IMAGE_PATH, model, transform, device)
    
    # Preview
    plt.imshow(depth, cmap='turbo')
    plt.title("Depth Map (Close=Yellow, Far=Purple)")
    plt.show()
    
    # Generate 3D model
    mesh, output_path = create_3d_model(depth, IMAGE_PATH)
    print(f"Model saved to: {output_path}")
    
    # Interactive view
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

if __name__ == "__main__":
    main()