import json


def write_cascades(cascade_ids, sequences, file_name):
    with open(file_name, 'w') as f:
        for cid in cascade_ids:
            cascade = sequences[cid]['cascade']
            # TODO: The first time is always 0. Try fetching the real times from db.
            rest = ' '.join(f'{item[0]} {item[1] * 3600.0 * 24 * 30}' for item in cascade[1:])
            line = f'{cascade[0][0]} {rest}\n'
            f.write(line)


def main():
    data_name = 'memetracker'

    with open(f'data/{data_name}/sequences.json') as f:
        sequences = json.load(f)

    with open(f'data/{data_name}/graph_info.json') as f:
        graph_info = json.load(f)

    with open(f'data/{data_name}/samples.json') as f:
        samples = json.load(f)

    training = graph_info['graph1']
    validation = list(set(samples['training']) - set(training))
    test = samples['test']

    write_cascades(training, sequences, f'data/{data_name}/train.txt')
    write_cascades(validation, sequences, f'data/{data_name}/val.txt')
    write_cascades(test, sequences, f'data/{data_name}/test.txt')


if __name__ == '__main__':
    main()
