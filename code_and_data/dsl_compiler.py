"""
This module contains code to compile and execute a subset of the DSL in `dsl.py`.
"""
import torch
import torch.nn.functional as F
from dsl import *
from near import init_optimizer
import pyparsing as pp
import pytorch_lightning as pl
from sklearn.metrics import f1_score

class CalMSLang:
    def build_lexer(self):
        lparen, rparen = pp.Suppress("("), pp.Suppress(")")
        nonterminal = pp.Word(pp.alphas, pp.alphanums + "_")
        terminal = pp.Word(pp.alphas, pp.alphanums + "_")
        self.expression = pp.Forward()
        self.expression <<= pp.Group(nonterminal + lparen + pp.delimitedList(self.expression) + rparen | terminal)

    def __init__(self, input_size=18, output_size=2, num_units=16, device="cpu"):
        self.input_size = input_size
        self.output_size = output_size
        self.num_units = num_units
        self.build_lexer()
        common_args = dict(input_size=input_size, output_size=output_size, num_units=num_units, device=device)
        self.grammar = {
            'RunningAvg': lambda map_fn : RunningAverageFunction(a2a_function=map_fn, **common_args),
            'Window5Avg': lambda map_fn : RunningAverageWindow5Function(a2a_function=map_fn, **common_args),
            'Window7Avg': lambda map_fn : RunningAverageWindow7Function(a2a_function=map_fn, **common_args),
            'Window13Avg': lambda map_fn : RunningAverageWindow13Function(a2a_function=map_fn, **common_args),            
            'Or': lambda fn1, fn2 : OrFunction(function1=fn1, function2=fn2, input_type='atom', output_type='atom', **common_args),
            'And': lambda fn1, fn2 : AndFunction(function1=fn1, function2=fn2, input_type='atom', output_type='atom', **common_args),
            "ResAngleHeadBodySelect" : lambda  : CALMS21ResAngleHeadBodySelection(**common_args),
            "SpeedSelect" : lambda  : CALMS21SpeedSelection(**common_args),
            "TangentialVelocitySelect" : lambda  : CALMS21TangentialVelocitySelection(**common_args),
            "AccelerationSelect" : lambda  : CALMS21AccelerationSelection(**common_args),
            "RelAngleSocialSelect" : lambda  : CALMS21RelAngleSocialSelection(**common_args),
            "AxisRatioSelect" : lambda  : CALMS21AxisRatioSelection(**common_args),
            "OverlapBboxesSelect" : lambda  : CALMS21OverlapBboxesSelection(**common_args),
            "MinResNoseKeypointDistSelect" : lambda  : CALMS21MinResNoseKeypointDistSelection(**common_args),
            "ListToAtomModule" : lambda  : ListToAtomModule(**common_args),
            "AtomToAtomModule" : lambda  : AtomToAtomModule(**common_args)
        }

    def parse(self, ast):        
        fn_name = ast[0]
        assert fn_name in self.grammar, f"function '{fn_name}' not found in {self.grammar.keys()}"
        fn_args = ast[1:]
        evaluated_fn_args = [self.parse(arg) for arg in fn_args]
        fn = self.grammar[fn_name]
        return fn(*evaluated_fn_args)
            

    def compile_program(self, program : str) -> LibraryFunction:
        """Compile a program into a LibraryFunction"""
        tree = self.expression.parseString(program)
        return self.parse(list(tree)[0])
    
class ExpertProgram(pl.LightningModule):
    def __init__(self, program : str, config):
        super().__init__()
        self.program = program
        self.config = config
        self.lang = CalMSLang(device=self.device)
        self.model = self.lang.compile_program(program)

    
    def forward(self, x):
        """
        x : (batch_size, 13, 18)
        output : (batch_size, num_classes)
        """
        output = self.model.execute_on_batch(x, batch_lens=[x.shape[1]] * x.shape[0])
        return output
    
    def f1_score(self, y_true, y_pred):
        """
        y_true : (batch_size, num_classes)
        y_pred : (batch_size, num_classes)
        """
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        return f1_score(y_true, y_pred, average="binary")

    def step(self, batch, stage=None):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.squeeze(1))

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", torch.eq(y_hat.argmax(-1), y).float().mean())
            self.log(f"{stage}_f1", self.f1_score(y, y_hat.argmax(-1)), prog_bar=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        nll_loss = self.step(batch, stage=None)
        self.log("train_loss", nll_loss)
        return nll_loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, stage="valid")

    def test_step(self, batch, batch_idx):
        self.step(batch, stage="test")

    def configure_optimizers(self):
        return init_optimizer(self.model, torch.optim.Adam, lr=self.config.lr)
