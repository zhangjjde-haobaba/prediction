from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import make_response
import io

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    #request.form：这是一个 ImmutableMultiDict 类型的字典，它包含了表单中所有的字段和它们的值。
    #如果你的表单是以 POST 方法提交的，你可以通过 request.form 来访问这些数据。
    # 例如，如果你有一个表单字段 <input type="text" name="username">，你可以通过 request.form['username'] 来获取用户输入的用户名。

    #request.files：这个属性用于访问上传的文件，它也是一个字典。
    # 当你的表单包含 <input type="file" name="file"> 这样的字段时，用户上传的文件可以通过 request.files['file'] 来获取。

    #request.args：用于访问 URL 参数（即查询字符串）。如果你的表单是以 GET 方法提交的，或者你想访问 URL 中的参数，
    # 可以使用 request.args。例如，对于 URL http://example.com?search=query，你可以使用 request.args.get('search') 来获取 search 参数的值。

    scaler = MinMaxScaler()
    # 添加模型加载和预测
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        df['main_focus'] = df['main_focus'].map({True: 1, False: 0})
        taskEncoder = OneHotEncoder(dtype=int)
        # 选择 'region' 列并对其进行独热编码
        task_region_encoded = taskEncoder.fit_transform(df[['region']]).toarray()
        task_columns_encoded = taskEncoder.get_feature_names_out(['region'])
        task_region_encoded_df = pd.DataFrame(task_region_encoded, columns=task_columns_encoded)
        # 合并独热编码后的 'region' 列和其他特征列，删除原始 'region' 列
        task = pd.concat([df.drop('region', axis=1), task_region_encoded_df], axis=1)
        # 列名要跟训练的一致
        train_columns = ['undesirable_event', 'feedback', 'positive_feedback', 'competitive', 'spread_rate',
                         'transfer_rate', 'matain_rate', 'main_focus', 'region_CC', 'region_EC', 'region_EN',
                         'region_NC', 'region_SC']

        # 在对 task 数据集进行预测之前，调整其列的顺序以匹配训练数据集
        task_prepared = task[train_columns]
        model = joblib.load('model/extra_tree_best_model.joblib')
        scaler = joblib.load('model/scaler.joblib')
        prediction = model.predict(task_prepared)
        # 将预测结果从归一化的范围转换回原始范围
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
        # 转化为整数
        predicted_sales_rounded = np.round(prediction).astype(int)
        # 将预测结果添加到 DataFrame 中
        task['sales_record'] = predicted_sales_rounded
        # 创建一个输出流
        output = io.StringIO()
        # 将 DataFrame 保存到 StringIO 对象
        task.to_csv(output, index=False)  # 设置 index=False 避免额外的索引列
        output.seek(0)  # 跳转到输出流的开头

        # 创建 Flask 响应对象
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=predicted_data.csv"
        response.headers["Content-type"] = "text/csv"
        return response
    else:
        return 'Invalid file type'



    # 重定向到新的页面或者重新渲染主页并显示预测结果
    # return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    #debug=true开启调试模式
    app.run(debug=True)
