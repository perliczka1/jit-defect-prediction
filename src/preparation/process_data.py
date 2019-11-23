LINE_ADDED_THRESHOLD = 0
START_DATE = {"qt": 1308350292,
              "openstack": 1322599384
              }

END_DATE = {"qt": 1395090476,
            "openstack": 1393590700
            }

CHURN_THRESHOLD = 10000
FILE_THRESHOLD = 100


def filter_as_in_jit_moving_target(df, project):
    df = df[df['churn'] <= CHURN_THRESHOLD]
    df = df[df['nf'] <= FILE_THRESHOLD]
    df = df[df['la'] > LINE_ADDED_THRESHOLD]
    df = df[df['author_date'] >= START_DATE[project]]
    df = df[df['author_date'] < END_DATE[project]]
    return df




