import os
from PIL import Image

# Directorios de imágenes y máscaras
TRAIN_IMG_DIR = './road_dataset/train_images'
TRAIN_MASK_DIR = './road_dataset/train_masks'
VAL_IMG_DIR = './road_dataset/val_images'
VAL_MASK_DIR = './road_dataset/val_masks'

# Función para verificar si una imagen está corrupta
def is_image_corrupted(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verifica si el archivo es válido
        return False  # La imagen no está corrupta
    except (IOError, SyntaxError):
        return True  # La imagen está corrupta

# Función para eliminar una imagen y su máscara correspondiente
def remove_image_and_mask(image_path, mask_dir):
    file_name = os.path.basename(image_path)  # Obtener solo el nombre del archivo
    mask_path = os.path.join(mask_dir, file_name)
    
    if os.path.exists(image_path):
        os.remove(image_path)  # Eliminar la imagen
        print(f"Imagen eliminada: {image_path}")
    
    if os.path.exists(mask_path):
        os.remove(mask_path)  # Eliminar la máscara
        print(f"Máscara eliminada: {mask_path}")
    
# Función para analizar un directorio de imágenes y eliminar imágenes corruptas
def analyze_and_clean_directory(image_dir, mask_dir):
    corrupted_files = []
    print(f"Analizando el directorio: {image_dir}")
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                if is_image_corrupted(file_path):
                    print(f"Imagen corrupta encontrada: {file_path}")
                    remove_image_and_mask(file_path, mask_dir)
                    corrupted_files.append(file_path)
    return corrupted_files

# Analizar todos los directorios de imágenes y máscaras
def main():
    directories = [
        (TRAIN_IMG_DIR, TRAIN_MASK_DIR),
        (VAL_IMG_DIR, VAL_MASK_DIR)
    ]
    
    all_corrupted = []
    
    for img_dir, mask_dir in directories:
        corrupted_files = analyze_and_clean_directory(img_dir, mask_dir)
        all_corrupted.extend(corrupted_files)
    
    # Resumen de resultados
    if all_corrupted:
        print("\nImágenes corruptas eliminadas:")
        for corrupted in all_corrupted:
            print(corrupted)
    else:
        print("\nNo se encontraron imágenes corruptas.")
    
if __name__ == '__main__':
    main()
