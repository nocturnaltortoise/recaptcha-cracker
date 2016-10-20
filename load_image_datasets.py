import tensorflow as tf
import glob

filenames = glob.glob('svhn-train/*.png')
filename_queue = tf.train.string_input_producer(filenames)

image_reader = tf.WholeFileReader()

images = []
for i in range(len(filenames)):
    print("Decoding image {image_no}/{total_image_count}"
          .format(image_no=i, total_image_count=len(filenames)))
    name, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_png(image_file)
    images.append(image)

# image_tensors = []
with tf.Session() as sess:
    # print("Initialising variables and setting up threads.")
    # sess.run(tf.initialize_all_variables())
    # think this is only necessary when we have placeholders or
    # anything else in tf's graph
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print("Image shape:")
    print(images[5].eval().shape)

    print("Stopping...")
    coord.request_stop()
    coord.join(threads)
