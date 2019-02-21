from ParlAI.projects.convai.convai_world import ConvAIWorld
from parlai.core.params import ParlaiParser
from agent import TransformerAgent


def main():
    parser = ParlaiParser(True, True)
    parser.set_defaults(batchsize=10,
                        sample=True,
                        clean_emoji=True,
                        check_grammar=True,
                        max_seq_len=256,
                        beam_size=3,
                        annealing_topk=None,
                        annealing=0.6,
                        length_penalty=0.7)

    ConvAIWorld.add_cmdline_args(parser)
    TransformerAgent.add_cmdline_args(parser)
    opt = parser.parse_args()

    agent = TransformerAgent(opt)
    world = ConvAIWorld(opt, [agent])

    while True:
        try:
            world.parley()
        except Exception as e:
            print('Exception: {}'.format(e))


if __name__ == '__main__':
    main()

