from json import loads, dumps
import argparse
from pathlib import Path
import regex as re

EOS = '<|endoftext|>'


def get_parser():
  parser = argparse.ArgumentParser(description='Remove EOS token and clip at stop words')
  parser.add_argument('--stop_words', type=str, required=True)
  parser.add_argument('--input_dir', type=str, required=True)
  parser.add_argument('--output_dir', type=str, required=True)
  return parser


def main(args):
  stop_words = args.stop_words.split(',')
  stop_words_regex = re.compile(rf"(.*?)\b(?:{'|'.join(stop_words)})\b.*", re.DOTALL)
  input_dir = Path(args.input_dir)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(exist_ok=True, parents=True)
  for file in input_dir.glob('*.jsonl'):
    with file.open() as f, (output_dir / file.name).open('w') as out:
      convert_count = 0
      for line in f:
        data = loads(line)
        for i in range(len(data['completion'])):
          converted = False
          completion = data['completion'][i]
          if completion.endswith(EOS):
            completion = completion[:-len(EOS)]
            converted = True
          if stop_words_regex.match(completion):
            completion = stop_words_regex.match(completion).group(1)
            converted = True
          data['completion'][i] = completion
          convert_count += int(converted)
        print(dumps(data), file=out)
      print(f'Converted {convert_count} completions in {file.name}')


if __name__ == '__main__':
  parser = get_parser()
  args = parser.parse_args()
  main(args)
