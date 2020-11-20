{
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0-final"
  },
  "orig_nbformat": 2,
  "kernelspec": {
   "name": "python3",
   "display_name": "Python 3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2,
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 177,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn import model_selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 178,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_raw = pd.read_csv('train.csv')\n",
    "test_raw = pd.read_csv('test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "    User_ID Product_ID Gender    Age  Occupation City_Category  \\\n",
       "1   1000001  P00248942      F   0-17          10             A   \n",
       "6   1000004  P00184942      M  46-50           7             B   \n",
       "13  1000005  P00145042      M  26-35          20             A   \n",
       "14  1000006  P00231342      F  51-55           9             A   \n",
       "16  1000006   P0096642      F  51-55           9             A   \n",
       "\n",
       "   Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "1                           2               0                   1   \n",
       "6                           2               1                   1   \n",
       "13                          1               1                   1   \n",
       "14                          1               0                   5   \n",
       "16                          1               0                   2   \n",
       "\n",
       "    Product_Category_2  Product_Category_3  Purchase  \n",
       "1                  6.0                14.0     15200  \n",
       "6                  8.0                17.0     19215  \n",
       "13                 2.0                 5.0     15665  \n",
       "14                 8.0                14.0      5378  \n",
       "16                 3.0                 4.0     13055  "
      ],
      "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>User_ID</th>\n      <th>Product_ID</th>\n      <th>Gender</th>\n      <th>Age</th>\n      <th>Occupation</th>\n      <th>City_Category</th>\n      <th>Stay_In_Current_City_Years</th>\n      <th>Marital_Status</th>\n      <th>Product_Category_1</th>\n      <th>Product_Category_2</th>\n      <th>Product_Category_3</th>\n      <th>Purchase</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>1</th>\n      <td>1000001</td>\n      <td>P00248942</td>\n      <td>F</td>\n      <td>0-17</td>\n      <td>10</td>\n      <td>A</td>\n      <td>2</td>\n      <td>0</td>\n      <td>1</td>\n      <td>6.0</td>\n      <td>14.0</td>\n      <td>15200</td>\n    </tr>\n    <tr>\n      <th>6</th>\n      <td>1000004</td>\n      <td>P00184942</td>\n      <td>M</td>\n      <td>46-50</td>\n      <td>7</td>\n      <td>B</td>\n      <td>2</td>\n      <td>1</td>\n      <td>1</td>\n      <td>8.0</td>\n      <td>17.0</td>\n      <td>19215</td>\n    </tr>\n    <tr>\n      <th>13</th>\n      <td>1000005</td>\n      <td>P00145042</td>\n      <td>M</td>\n      <td>26-35</td>\n      <td>20</td>\n      <td>A</td>\n      <td>1</td>\n      <td>1</td>\n      <td>1</td>\n      <td>2.0</td>\n      <td>5.0</td>\n      <td>15665</td>\n    </tr>\n    <tr>\n      <th>14</th>\n      <td>1000006</td>\n      <td>P00231342</td>\n      <td>F</td>\n      <td>51-55</td>\n      <td>9</td>\n      <td>A</td>\n      <td>1</td>\n      <td>0</td>\n      <td>5</td>\n      <td>8.0</td>\n      <td>14.0</td>\n      <td>5378</td>\n    </tr>\n    <tr>\n      <th>16</th>\n      <td>1000006</td>\n      <td>P0096642</td>\n      <td>F</td>\n      <td>51-55</td>\n      <td>9</td>\n      <td>A</td>\n      <td>1</td>\n      <td>0</td>\n      <td>2</td>\n      <td>3.0</td>\n      <td>4.0</td>\n      <td>13055</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
     },
     "metadata": {},
     "execution_count": 179
    }
   ],
   "source": [
    "train = train_raw.dropna()\n",
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "metadata": {},
   "outputs": [
    {
     "output_type": "execute_result",
     "data": {
      "text/plain": [
       "   User_ID Product_ID Gender    Age  Occupation City_Category  \\\n",
       "4  1000011  P00053842      F  26-35           1             C   \n",
       "5  1000013  P00350442      M  46-50           1             C   \n",
       "6  1000013  P00155442      M  46-50           1             C   \n",
       "7  1000013   P0094542      M  46-50           1             C   \n",
       "8  1000015  P00161842      M  26-35           7             A   \n",
       "\n",
       "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "4                          1               0                   4   \n",
       "5                          3               1                   2   \n",
       "6                          3               1                   1   \n",
       "7                          3               1                   2   \n",
       "8                          1               0                  10   \n",
       "\n",
       "   Product_Category_2  Product_Category_3  \n",
       "4                 5.0                12.0  \n",
       "5                 3.0                15.0  \n",
       "6                11.0                15.0  \n",
       "7                 4.0                 9.0  \n",
       "8                13.0                16.0  "
      ],
      "text/html": "<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>User_ID</th>\n      <th>Product_ID</th>\n      <th>Gender</th>\n      <th>Age</th>\n      <th>Occupation</th>\n      <th>City_Category</th>\n      <th>Stay_In_Current_City_Years</th>\n      <th>Marital_Status</th>\n      <th>Product_Category_1</th>\n      <th>Product_Category_2</th>\n      <th>Product_Category_3</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>4</th>\n      <td>1000011</td>\n      <td>P00053842</td>\n      <td>F</td>\n      <td>26-35</td>\n      <td>1</td>\n      <td>C</td>\n      <td>1</td>\n      <td>0</td>\n      <td>4</td>\n      <td>5.0</td>\n      <td>12.0</td>\n    </tr>\n    <tr>\n      <th>5</th>\n      <td>1000013</td>\n      <td>P00350442</td>\n      <td>M</td>\n      <td>46-50</td>\n      <td>1</td>\n      <td>C</td>\n      <td>3</td>\n      <td>1</td>\n      <td>2</td>\n      <td>3.0</td>\n      <td>15.0</td>\n    </tr>\n    <tr>\n      <th>6</th>\n      <td>1000013</td>\n      <td>P00155442</td>\n      <td>M</td>\n      <td>46-50</td>\n      <td>1</td>\n      <td>C</td>\n      <td>3</td>\n      <td>1</td>\n      <td>1</td>\n      <td>11.0</td>\n      <td>15.0</td>\n    </tr>\n    <tr>\n      <th>7</th>\n      <td>1000013</td>\n      <td>P0094542</td>\n      <td>M</td>\n      <td>46-50</td>\n      <td>1</td>\n      <td>C</td>\n      <td>3</td>\n      <td>1</td>\n      <td>2</td>\n      <td>4.0</td>\n      <td>9.0</td>\n    </tr>\n    <tr>\n      <th>8</th>\n      <td>1000015</td>\n      <td>P00161842</td>\n      <td>M</td>\n      <td>26-35</td>\n      <td>7</td>\n      <td>A</td>\n      <td>1</td>\n      <td>0</td>\n      <td>10</td>\n      <td>13.0</td>\n      <td>16.0</td>\n    </tr>\n  </tbody>\n</table>\n</div>"
     },
     "metadata": {},
     "execution_count": 180
    }
   ],
   "source": [
    "test = test_raw.dropna()\n",
    "test.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_features = train.select_dtypes(include=['int64', 'float64']).columns\n",
    "categorical_features = train.select_dtypes(include=['object']).columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "metadata": {},
   "outputs": [],
   "source": [
    "numeric_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='median')),\n",
    "    ('scaler', StandardScaler())])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "metadata": {},
   "outputs": [],
   "source": [
    "categorical_transformer = Pipeline(steps=[\n",
    "    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),\n",
    "    ('onehot', OneHotEncoder(handle_unknown='ignore'))])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "metadata": {},
   "outputs": [],
   "source": [
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', numeric_transformer, numeric_features),\n",
    "        ('cat', categorical_transformer, categorical_features)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "metadata": {},
   "outputs": [],
   "source": [
    "Y = train['Purchase']\n",
    "X = train\n",
    "X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 186,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr = LinearRegression()\n",
    "lr = Pipeline(steps=[('preprocessor', preprocessor),\n",
    "                      ('classifier', LinearRegression())])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "metadata": {},
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": [
      "model score: -0.993\n"
     ]
    }
   ],
   "source": [
    "lr.fit(X_train, Y_train)\n",
    "print(\"model score: %.3f\" % lr.score(X_test, y_test))"
   ]
  }
 ]
}