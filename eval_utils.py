def calculate_metrics(data):
    # 初始化统计变量
    pre_metrics = {
        "rewrite_acc": 0.0,
        "rephrase_acc": 0.0,
        "locality_acc": 0.0,
        "portability_acc": 0.0
    }
    post_metrics = {
        "rewrite_acc": 0.0,
        "rephrase_acc": 0.0,
        "locality_acc": 0.0,
        "portability_acc": 0.0
    }
    
    total_cases = len(data)
    
    # 用于统计所有 case 的各项均值
    total_pre_metrics = {
        "rewrite_acc": 0.0,
        "rephrase_acc": 0.0,
        "locality_acc": 0.0,
        "portability_acc": 0.0
    }
    total_post_metrics = {
        "rewrite_acc": 0.0,
        "rephrase_acc": 0.0,
        "locality_acc": 0.0,
        "portability_acc": 0.0
    }
    
    # 遍历所有 case，计算每个 case 的 pre 和 post 部分的指标
    for case in data:
        # 处理 pre 部分
        pre = case.get("pre", {})
        if "rewrite_acc" in pre:
            pre_metrics["rewrite_acc"] = sum(pre["rewrite_acc"]) / len(pre["rewrite_acc"]) if pre["rewrite_acc"] else 0.0
        if "rephrase_acc" in pre:
            pre_metrics["rephrase_acc"] = sum(pre["rephrase_acc"]) / len(pre["rephrase_acc"]) if pre["rephrase_acc"] else 0.0
        if "locality" in pre and "neighborhood_acc" in pre["locality"]:
            pre_metrics["locality_acc"] = sum(pre["locality"]["neighborhood_acc"]) / len(pre["locality"]["neighborhood_acc"]) if pre["locality"]["neighborhood_acc"] else 0.0
        # if "portability" in pre:
            # pre_metrics["portability_acc"] = sum(pre["portability"]["neighborhood_acc"]) / len(pre["portability"]["neighborhood_acc"]) if pre["portability"]["neighborhood_acc"] else 0.0
        
        # 累加到总的 pre 指标
        total_pre_metrics["rewrite_acc"] += pre_metrics["rewrite_acc"]
        total_pre_metrics["rephrase_acc"] += pre_metrics["rephrase_acc"]
        total_pre_metrics["locality_acc"] += pre_metrics["locality_acc"]
        total_pre_metrics["portability_acc"] += pre_metrics["portability_acc"]
        
        # 处理 post 部分
        post = case.get("post", {})
        if "rewrite_acc" in post:
            post_metrics["rewrite_acc"] = sum(post["rewrite_acc"]) / len(post["rewrite_acc"]) if post["rewrite_acc"] else 0.0
        if "rephrase_acc" in post:
            post_metrics["rephrase_acc"] = sum(post["rephrase_acc"]) / len(post["rephrase_acc"]) if post["rephrase_acc"] else 0.0
        if "locality" in post and "neighborhood_acc" in post["locality"]:
            post_metrics["locality_acc"] = sum(post["locality"]["neighborhood_acc"]) / len(post["locality"]["neighborhood_acc"]) if post["locality"]["neighborhood_acc"] else 0.0
        # if "portability" in post:
            # post_metrics["portability_acc"] = sum(post["portability"]["neighborhood_acc"]) / len(post["portability"]["neighborhood_acc"]) if post["portability"]["neighborhood_acc"] else 0.0
        
        # 累加到总的 post 指标
        total_post_metrics["rewrite_acc"] += post_metrics["rewrite_acc"]
        total_post_metrics["rephrase_acc"] += post_metrics["rephrase_acc"]
        total_post_metrics["locality_acc"] += post_metrics["locality_acc"]
        total_post_metrics["portability_acc"] += post_metrics["portability_acc"]
    
    # 计算所有 case 的均值
    average_pre_metrics = {key: total_pre_metrics[key] / total_cases for key in total_pre_metrics}
    average_post_metrics = {key: total_post_metrics[key] / total_cases for key in total_post_metrics}
    
    return average_pre_metrics, average_post_metrics
