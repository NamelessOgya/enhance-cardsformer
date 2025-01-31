from datetime import datetime, timedelta


def get_experiment_code():
    # 現在のUTC時間を取得
    now_utc = datetime.utcnow()

    # 9時間を加えてJSTを計算
    now_jst = now_utc + timedelta(hours=9)

    # 指定フォーマットで文字列に変換
    formatted_time = now_jst.strftime("%Y%m%d%H%M%S")

    return formatted_time