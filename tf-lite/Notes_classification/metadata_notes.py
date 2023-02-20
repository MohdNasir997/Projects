from tflite_support.metadata_writers import image_classifier
from tflite_support.metadata_writers import writer_utils
ImageClassifierwriter = image_classifier.MetadataWriter
model_path = '/home/nasir/Desktop/dataset/notes.tflite'
lables_path = '/home/nasir/Desktop/dataset/labels.txt'
saved_path = '/home/nasir/Desktop/dataset/notes-metadata.tflite'
input_norm_mean = 127.5
input_norm_std = 127.5
writer = ImageClassifierwriter.create_for_inference(writer_utils.load_file(model_path),[input_norm_mean],[input_norm_std],[lables_path])

print(writer.get_metadata_json())
writer_utils.save_file(writer.populate(),saved_path)