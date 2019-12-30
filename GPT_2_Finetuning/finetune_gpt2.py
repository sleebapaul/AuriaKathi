from gpt_2 import finetune, start_tf_sess
import argparse

if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description='TensorFlow Haiku GPT-2 Finetuned Language Model')
    parser.add_argument('--dataset', type=str,
                        help='location of the data corpus')
    parser.add_argument('--model_name', type=str, default='models/117M',
                        help='Pretrained model path')
    parser.add_argument('--steps', type=int, default=1000,
                        help='No of epochs to train during finetuning')
    args = parser.parse_args()
    sess = start_tf_sess()
    finetune(sess,
              dataset=args.dataset,
              model_name=args.model_name,
              steps=args.steps,
              restore_from='fresh',
              run_name='run1',
              print_every=100,
              sample_every=500,
              save_every=500
              )
    print("Finetuning completed!!!")
 