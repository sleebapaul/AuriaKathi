import argparse
import os
import random

from gpt_2 import generateAML, load_gpt2, start_tf_sess

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='TensorFlow Haiku GPT-2 generation form trained Language Model')
    parser.add_argument('--run_name', type=str,
                        help='Run number of training')
    parser.add_argument('--output_folder', type=str,
                        help='Output folder for saving haiku.txt')
    args = parser.parse_args()
    sess = start_tf_sess()
    load_gpt2(sess, run_name=args.run_name)
    text = generateAML(sess,
              run_name=args.run_name
              )
    text = text[0].split("\n\n")
    text.pop(0)
    random_haiku = random.choice(text)
    if not (args.output_folder is None):
        os.makedirs(args.output_folder, exist_ok=True)
    print("{} created".format(args.output_folder))
    with open('{}/haiku.txt'.format(args.output_folder), 'w+') as f:
        f.write(random_haiku)
    print("Haiku is generated!")

    print(os.path.dirname(os.path.realpath(__file__)))
