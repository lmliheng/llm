from laya import Router

router = Router(device="cpu", preload=True)

state = {"text": "用户说：帮我退款，重复扣费了"}
questions = {
    "intent": {
        "type": "choice",
        "instructions": "判断用户意图",
        "criteria": {
            "refund": "退款/扣费问题",
            "tech": "技术故障",
            "sales": "销售咨询",
            "other": "其他"
        }
    }
}

print(router.predict(state, questions))