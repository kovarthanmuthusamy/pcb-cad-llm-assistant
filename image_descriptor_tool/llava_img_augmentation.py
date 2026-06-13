import os
import json
import cv2
import albumentations as A
from tqdm import tqdm

if __name__ == "__main__":
    
    #List all the augmentations to be done
    augmentations = {
        "horizontal_flip": A.HorizontalFlip(p=1.0),
        "vertical_flip": A.VerticalFlip(p=1.0),
        #"rotate_90": A.Rotate(limit=(90, 90), p=1.0),
        "rotate_180": A.Rotate(limit=(180, 180), p=1.0),
        #"rotate_270": A.Rotate(limit=(270, 270), p=1.0),
        "plasme_brightness_contrast": A.PlasmaBrightnessContrast(p=1.0),
        "gaussian_noise": A.GaussNoise(p=1.0),
        "rgb_shift": A.RGBShift(p=1.0),
        "channel_shuffle": A.ChannelShuffle(p=1.0),
        "emboss": A.Emboss(p=1.0),
        "median_blur": A.MedianBlur(p=1.0),
        "color_jitter": A.ColorJitter(p=1.0),
        "sepia": A.ToSepia(p=1.0),
        "auto_contrast": A.AutoContrast(p=1.0)
    }

    orig_dataset_file_name = "json_data_llava_01_02_2025_rephrased_text_aug.json"

    #Open the original dataset (unaugmented)
    with open(orig_dataset_file_name, "r") as file:
        dataset = json.load(file)

    new_entries = []
    next_id = len(dataset) + 1
    

    for entry in tqdm(dataset, desc="Processing dataset....."):
        
        image_name = os.path.basename(entry["image"]) #Gets the image name from the dataset
        image_path_folders = os.path.normpath(entry["image"]).split(os.sep) #Splits the path by '/' and gets the names of the folders as a list (The first element would be the base folder)
                                                                            #normpath is used to make the '/' to '\' to make it compatible with windows or vice versa

        if not os.path.exists(entry["image"]):
            print(f"Image not found: {image_name}")
            continue
        
        augmented_paths = []

        for aug_name, aug_transform in augmentations.items():   #Augmentations application loop
            img = cv2.imread(entry["image"])
            augmented_img = aug_transform(image=img)["image"]
            img_base_path = os.path.splitext(os.path.basename(entry['image']))[0] 
            
            #Create a subfolder with the name of the augmentation
            new_subfolder_path = os.path.join(image_path_folders[0], aug_name) 
            os.makedirs(new_subfolder_path, exist_ok=True)
            new_path = os.path.join(new_subfolder_path, f"{os.path.basename(os.path.splitext(entry['image'])[0])}_{aug_name}.png")

            cv2.imwrite(new_path, augmented_img)
            augmented_paths.append((aug_name, new_path))

        for augmentation, new_path in augmented_paths:  #Create the new dataset
            new_entry = entry.copy()
            new_entry["id"] = str(next_id)
            new_entry["image"] = new_path.replace("\\","/")
            new_entries.append(new_entry)
            next_id += 1
        
        augmented_dataset = dataset + new_entries #This appends the new augmented data to the previous data from dataset

    #Create a dataset and write the augmented data into the new file
    augmented_dataset_file = os.path.join(f"{os.path.splitext(orig_dataset_file_name)[0]}_img_augmented.json")
    

    with open(augmented_dataset_file, "w") as file_aug:
        json.dump(augmented_dataset, file_aug, indent=4)

    print(f"Augmented dataset saved to {augmented_dataset_file}")
