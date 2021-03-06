import csv

def read_csv(filepath, delimiter, skip_rows=0):
    '''
    default: can't filter the csv head
    :param filepath:
    :param delimiter:
    :return:
    '''
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = csv.reader(f, delimiter=delimiter)
        lines = [line for line in lines]
        return lines[skip_rows:]

def write_csv(filepath, data, delimiter):
    '''
    TSV is Tab-separated values and CSV, Comma-separated values
    :param data, is list
    '''
    with open(filepath, 'w', encoding='utf-8') as f:
        csv_writer = csv.writer(f, delimiter=delimiter)
        csv_writer.writerows(data)

def get_mask_template(bert_data_filepath, output_data_filepath, temp):
    label_map = {0: 'anger', 1: 'happy', 2:'neutral', 3:'sad'}
    new_instances = []
    instances = read_csv(bert_data_filepath, delimiter=',')
    new_instances.append(instances[0])
    for instance in instances[1:]:
        label, text = instance
        label_name = label_map[int(label)]
        if text[-1] not in ['?', '!', '.']:
            text += '.'
        text += temp
        new_instances.append([label_name, text])
    print('all instances {}'.format(len(new_instances)))
    write_csv(output_data_filepath, new_instances, delimiter=',')

def get_nsp_template(bert_data_filepath, output_data_filepath, temp=' I am'):
    label_map = {0: 'anger', 1: 'happy', 2:'neutral', 3:'sad'}
    new_instances = []
    instances = read_csv(bert_data_filepath, delimiter=',')
    new_instances.append(instances[0] + ['sentence2'])
    for instance in instances[1:]:
        label, text = instance
        label_name = label_map[int(label)]
        for label in label_map.values():
            text2 = temp.replace('[MASK]', label)
            if label_name == label:
                target = 1
            else:
                target = 0
            new_instances.append([target, text, text2])
    print('all instances {}'.format(len(new_instances)))
    write_csv(output_data_filepath, new_instances, delimiter=',')


if __name__ == '__main__':
    set_name = 'tst'
    for cv_no in range(11, 13):
        template = [
            ' i am [MASK] .',
            ' it was [MASK] .',
            " i'm [MASK].",
            ' overall, it expresses [MASK] emotion .',
            ' it is [MASK] .',
            ' [MASK] .',
            ' [MASK] emotion .',
            ' he is [MASK]'
        ][7]
        template_name = template.replace('[MASK]', '').replace('.', '').replace("'", '-').replace(',', '').replace(' ', '')
        #### for iemocap 
        # bert_data_filepath = '/data7/emobert/exp/evaluation/IEMOCAP/bert_data/{}/{}.csv'.format(cv_no, set_name)
        # output_data_filepath = '/data7/emobert/exp/promote_pretrain/data/iemocap/{}/{}_mask_{}.csv'.format(cv_no, set_name, template_name)
        # get_mask_template(bert_data_filepath, output_data_filepath, temp=template)
        # output_data_filepath = '/data7/emobert/exp/promote_pretrain/data/iemocap/{}/{}_nsp_{}.csv'.format(cv_no, set_name, template_name)
        # get_nsp_template(bert_data_filepath, output_data_filepath, temp=template)
        ### for msp
        bert_data_filepath = '/data7/emobert/exp/evaluation/MSP/bert_data/{}/{}.csv'.format(cv_no, set_name)
        output_data_filepath = '/data7/emobert/exp/promote_pretrain/data/msp/{}/{}_mask_{}.csv'.format(cv_no, set_name, template_name)
        get_mask_template(bert_data_filepath, output_data_filepath, temp=template)
        output_data_filepath = '/data7/emobert/exp/promote_pretrain/data/msp/{}/{}_nsp_{}.csv'.format(cv_no, set_name, template_name)
        get_nsp_template(bert_data_filepath, output_data_filepath, temp=template)