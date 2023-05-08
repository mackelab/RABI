from rbibm.metric.base import RobustnessMetric


class GlobalLipschitzBound(RobustnessMetric):
    def eval(self, name="embedding_net", norm="l2"):
        pass


class IBPLocalLipschitzBound(RobustnessMetric):
    def eval(self, name="embedding_net", norm="l2"):
        pass


class IBPLossBound(RobustnessMetric):
    def eval(self, *args, **kwargs):
        return super().eval(*args, **kwargs)
