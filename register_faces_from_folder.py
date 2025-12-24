"""
Batch Face Registration
=======================
Register multiple faces from a folder structure.

Expected folder structure:
    known_faces/
        person1/
            image1.jpg
            image2.jpg
        person2/
            image1.jpg
        ...

The folder name becomes the person's name.
"""

import os
import sys
from face_recognition_system import FaceRecognitionSystem


def register_faces_from_folder(folder_path, system=None):
    """
    Register faces from a folder structure.
    
    Args:
        folder_path: Path to the folder containing person subfolders
        system: FaceRecognitionSystem instance (creates new one if None)
        
    Returns:
        dict: Statistics about registration
    """
    if system is None:
        system = FaceRecognitionSystem()
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return None
    
    stats = {
        'total_images': 0,
        'successful': 0,
        'failed': 0,
        'persons': []
    }
    
    # Iterate through person folders
    for person_name in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_name)
        
        if not os.path.isdir(person_path):
            continue
        
        print(f"\nProcessing: {person_name}")
        person_success = 0
        
        # Process each image in the person's folder
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            
            # Check if it's an image file
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                continue
            
            stats['total_images'] += 1
            
            if system.register_face_from_image(image_path, person_name):
                stats['successful'] += 1
                person_success += 1
            else:
                stats['failed'] += 1
        
        if person_success > 0:
            stats['persons'].append(person_name)
            print(f"  Registered {person_success} image(s) for {person_name}")
    
    print("\n" + "="*50)
    print("REGISTRATION SUMMARY")
    print("="*50)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Successful registrations: {stats['successful']}")
    print(f"Failed registrations: {stats['failed']}")
    print(f"Persons registered: {len(stats['persons'])}")
    
    return stats


def main():
    """Main function."""
    # Default folder path
    default_folder = "known_faces"
    
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input(f"Enter folder path (or press Enter for '{default_folder}'): ").strip()
        if not folder_path:
            folder_path = default_folder
    
    # Create example folder structure if it doesn't exist
    if not os.path.exists(folder_path):
        print(f"\nFolder '{folder_path}' not found.")
        create = input("Create example folder structure? (y/n): ").strip().lower()
        
        if create == 'y':
            os.makedirs(os.path.join(folder_path, "example_person"))
            print(f"\nCreated folder structure at: {folder_path}")
            print("Please add images to the person subfolders and run again.")
            print("\nExpected structure:")
            print(f"  {folder_path}/")
            print(f"    person1/")
            print(f"      image1.jpg")
            print(f"      image2.jpg")
            print(f"    person2/")
            print(f"      image1.jpg")
        return
    
    register_faces_from_folder(folder_path)


if __name__ == "__main__":
    main()
