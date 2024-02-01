from sklearn.model_selection import train_test_split
from evalvalid import *
from LogisticRegression.LogisticRegression import *
import PrepareData.preprocess as pp
import pandas as pd

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
def main():
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    train_ids, train_data, train_sc = pp.prepare_data(train_data)
    test_ids, test_data, _ = pp.prepare_data(test_data, train_sc)

    X = train_data.drop(columns=['Exited'])
    y = train_data['Exited']

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 0, test_size = 0.2)

    model = My_LogisticRegression(n_iter = 500, lr = 0.2,
                                  lambda_ = 0, err = True,
                                  debug = True, batch_size = X_train.shape[0]//10000)
    model.fit(X_train, y_train, X_val, y_val)

    y_pred = model.predict_proba(X_val)

    eval_valid(y_pred, y_val)

    plt.plot(model.errIN[15:], label = "IN")
    plt.plot(model.errOUT[15:], label = "OUT")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()