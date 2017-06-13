import cv2, os, json
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET


class Gallery:
    def __init__(self, path, gallery):
        self.path = path
        self.gallery = gallery

    def get_images_labels_and_gal(self):
        image_paths = [os.path.join(self.path, f) for f in os.listdir(self.path) if not f.endswith('.sad') and not f.startswith('.')]
        images = []
        labels = []
        new_image_found = False
        new_images_labels_returned = False
        person_ID_to_gallery = self.gallery
        for image_path in image_paths:
            person_ID = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            if not person_ID in person_ID_to_gallery:
                person_ID_to_gallery.setdefault(person_ID,[])
                new_image_found = True
            elif not image_path in [x for v in person_ID_to_gallery.values() for x in v]:
                new_image_found = True
            if new_image_found == True:
                person_ID_to_gallery[person_ID].append(image_path)
                image_pil = Image.open(image_path).convert('L')
                image = np.array(image_pil, 'uint8')
                images.append(image)
                labels.append(person_ID)
                displayer = display_image()
                displayer.display("Adding faces to training set..", image)
                displayer.remove_displays(50)
                new_image_found = False
        if images == [] and labels == []:
            print("Found no new images to add")
        else:
            new_images_labels_returned = True
        return images, labels, person_ID_to_gallery, new_images_labels_returned

class display_image:
    def __init__(self):
        cv2.startWindowThread()
    
    def display(self, name, image):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 300, 250)
        cv2.imshow(name, image)
    
    def remove_displays(self, *args):
        time = 0
        for x in args:
            time = x
        cv2.waitKey(time)
        cv2.destroyAllWindows()

class read_write_json:
    def __init__(self, path):
        self.path = path

    def write_to_file(self, gallery):
        with open(self.path, 'w') as f:
            json.dump(gallery, f)
        
    def read_from_file(self):
        gallery = {}
        with open(self.path, 'r') as f:
            try:
                gallery = json.load(f)
                gallery = {int(k):v for k,v in gallery.items()}
            except ValueError:
                print("No images in Gallery")
        return gallery
    
class probe:
    def __init__(self, path, gallery, recognizer):
        self.path = path
        self.gallery = gallery
        self.recognizer = recognizer

    def test_image_against_DB(self):
        image_paths = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith('.sad')]
        for image_path in image_paths:
            image_pil_probe = Image.open(image_path).convert('L')
            image_probe = np.array(image_pil_probe, 'uint8')
            person_ID_predicted, confidence = self.recognizer.predict(image_probe)
            link_to_gallery_image = self.gallery.get(person_ID_predicted, "ERROR: Not in gallery")[0]
            gallery_image = np.array(Image.open(link_to_gallery_image).convert('L'), 'uint8')
            person_ID_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            displayer = display_image()
            displayer.display("Person {}".format(person_ID_actual), image_probe)
            if person_ID_actual == person_ID_predicted and confidence < 30:
                print "Person {} is correctly recognized with confidence {}".format(person_ID_actual, confidence)
                displayer.display("Recognized as the following:", gallery_image)
            else:
                print "Person {} was not recognized. Closest match was to person {} with confidence {}".format(person_ID_actual, person_ID_predicted, confidence)
            displayer.remove_displays(1000)

def main():
    new_image_path = './images'
    gallery_path = 'gallery.json'
    r_w = read_write_json(gallery_path)
    gallery = r_w.read_from_file()
    existing_xml = False
    new_images_to_train = False
    recognizer = cv2.face.createLBPHFaceRecognizer()
    try :
        tree = ET.parse('recognizer.xml')
        root = tree.getroot()
        if root.tag == 'opencv_storage':
            print("Found existing face recognizer training file")
            existing_xml = True
            recognizer.load('recognizer.xml')
    except BaseException:
        print("No existing XML recognizer found")
    g = Gallery(new_image_path, gallery)
    images, labels, new_gallery, new_images_to_train = g.get_images_labels_and_gal()
    r_w.write_to_file(gallery)
    if new_images_to_train == True:
        print("Training or updating recognizer")
        if existing_xml == False:
            recognizer.train(images, np.array(labels))
        else:
            recognizer.update(images, np.array(labels))
    p = probe(new_image_path, new_gallery, recognizer)
    p.test_image_against_DB()
    recognizer.save('recognizer.xml')

if __name__ == "__main__":
    main()