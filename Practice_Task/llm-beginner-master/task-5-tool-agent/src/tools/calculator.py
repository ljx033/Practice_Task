"""calculator: 安全的数学表达式求值（AST 白名单，不裸 eval）。"""
import ast
import math
import operator

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": (
            "计算数学表达式，支持 + - * / // % ** 以及 sqrt/sin/cos/tan/"
            "log/log2/log10/exp/floor/ceil/round/abs/min/max/factorial 等函数，"
            "支持 pi / e / tau 常量。只接受纯数学表达式，不接受代码。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如 'sqrt(2026)'、'(123 + 456) * 789'、'round(sqrt(2026), 6)'。",
                }
            },
            "required": ["expression"],
        },
    },
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_SAFE_NAMES = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "pow": pow, "len": len, "int": int, "float": float,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "log": math.log, "log2": math.log2, "log10": math.log10, "exp": math.exp,
    "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "factorial": math.factorial, "gcd": math.gcd, "hypot": math.hypot,
    "degrees": math.degrees, "radians": math.radians,
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
    "True": True, "False": False, "true": True, "false": False,
}


class _SafeEvaluator(ast.NodeVisitor):
    def visit(self, node):
        method = "visit_" + type(node).__name__
        visitor = getattr(self, method, None)
        if visitor is None:
            raise ValueError(f"不支持的语法：{type(node).__name__}")
        return visitor(node)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ValueError(f"不支持的常量类型：{type(node.value).__name__}")

    def visit_Num(self, node):  # 兼容旧 Python AST
        return node.n

    def visit_Name(self, node):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]
        raise ValueError(f"未知名称：{node.id}")

    def visit_BinOp(self, node):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"不支持的运算符：{op_type.__name__}")
        return _BIN_OPS[op_type](self.visit(node.left), self.visit(node.right))

    def visit_UnaryOp(self, node):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"不支持的一元运算符：{op_type.__name__}")
        return _UNARY_OPS[op_type](self.visit(node.operand))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("只允许直接调用白名单函数")
        if node.func.id not in _SAFE_NAMES:
            raise ValueError(f"不允许调用：{node.func.id}")
        func = _SAFE_NAMES[node.func.id]
        if not callable(func):
            raise ValueError(f"{node.func.id} 不是函数")
        args = [self.visit(a) for a in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords if kw.arg}
        return func(*args, **kwargs)

    def visit_Tuple(self, node):
        return tuple(self.visit(e) for e in node.elts)

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]


def _fmt(x) -> str:
    """统一数字格式：整数原样，浮点最多 6 位小数并去掉尾零。
    这样 sqrt(2026)=45.01110973969615 → '45.01111'，能被关键词 '45.01111' 命中。"""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if x == int(x) and abs(x) < 1e16:
            return str(int(x))
        s = f"{x:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(x)


def run(args: dict) -> str:
    if not isinstance(args, dict) or "expression" not in args:
        raise KeyError("calculator 需要参数 {'expression': str}")
    expr = str(args["expression"]).strip()
    if not expr:
        raise ValueError("表达式为空")
    if len(expr) > 500:
        raise ValueError("表达式过长（>500 字符）")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"表达式语法错误：{e.msg}")
    value = _SafeEvaluator().visit(tree)
    return f"{expr} = {_fmt(value)}"