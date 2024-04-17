from tools.timetamp import add_timestamp


def get_csv_path(args):
    timestamp = add_timestamp()
    csv_path = ('./result/csv/' + str(args['model']) + '-' + str(args['dataset'])
                + '-' + str(args['attack_method']) + '-' + str(args['aggregate_function'])
                + '-malicious_rate:' + str(args['malicious_user_rate']) + '-epochs:'
                + str(args['epochs']) + timestamp + '.csv')
    return csv_path
