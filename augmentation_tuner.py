import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import sys
import os

def run_tuning_session(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] No se encuentra la imagen: {image_path}")
        return

    # Cargar la imagen original en formato PIL RGB
    original_img = Image.open(image_path).convert('RGB')

# =================================================================
    # HIPERPARÁMETROS DE COLOR JITTER (Stress Test)
    # =================================================================
    # Llevamos el brillo al +/- 70% y el contraste al +/- 60%
    jitter_transform = T.ColorJitter(
        brightness=0.7,  # Puede oscurecer hasta casi negro o sobreexponer a casi blanco
        contrast=0.6,    # Fuerza la diferencia entre la línea blanca y el piso gris
        saturation=0.5,  # Lava o satura agresivamente los colores
        hue=0.05         # Mantenemos el matiz casi intacto para no volver violeta el asfalto
    )

    # Configurar el lienzo (Grid de 4x4)
    fig, axes = plt.subplots(4, 4, figsize=(12, 10))
    fig.suptitle('Color Jittering Tuning Session (PyTorch Pipeline)', fontsize=16)

    # Posición [0,0]: Imagen Original
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')

    # Generar 15 variaciones estocásticas
    for i in range(1, 16):
        row = i // 4
        col = i % 4
        
        # Aplicar la transformación aleatoria
        aug_img = jitter_transform(original_img)
        
        axes[row, col].imshow(aug_img)
        axes[row, col].set_title(f"Variation {i}")
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig('tuning_grid.png', dpi=300)
    print("Grid generado: 'tuning_grid.png'. Ábrelo y analiza las variaciones.")
    plt.show()

if __name__ == "__main__":
    # Puedes pasar la ruta por consola o cambiar el string aquí directamente.
    # Ejemplo de uso desde consola: python augmentation_tuner.py DataSet/Record_2025-08-08_19-36-02/Images/1754674834877.png
    
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        # Reemplaza esto con una ruta válida real que saques de tu dataset_global.csv
        target_image = "DataSet/Record_2025-08-08_19-36-02/Images/1754674834877.png" 
        
    run_tuning_session(target_image)