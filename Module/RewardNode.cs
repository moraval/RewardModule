using System;
using System.ComponentModel;
using System.Linq;

using GoodAI.Core;
using GoodAI.Core.Memory;
using GoodAI.Core.Nodes;
using GoodAI.Core.Task;
using GoodAI.Core.Utils;
using ManagedCuda;
using YAXLib;
using GoodAI.Modules.GameBoy;

namespace RewardModule
{
    /// <author>An Engineer</author>
    /// <status>Working sample</status>
    /// <summary>Node for calculating reward for the EvolutionNode.</summary>
    /// <description>
    /// Node calculates reward for different reinforcement learning/classification tasks.
    /// </description>
    public class RewardNode : MyWorkingNode
    {
        [MyInputBlock(0)]
        public MyMemoryBlock<float> InputAgent
        {
            get { return GetInput(0); }
        }

        [MyInputBlock(1)]
        public MyMemoryBlock<float> InputTarget
        {
            get { return GetInput(1); }
        }

        [MyInputBlock(2)]
        public MyMemoryBlock<float> InputEvent
        {
            get { return GetInput(2); }
        }
        //reward for evolution
        [MyOutputBlock(0)]
        public MyMemoryBlock<float> Output
        {
            get { return GetOutput(0); }
            set { SetOutput(0, value); }
        }


        [MyBrowsable, Category("Number of Steps to reward"), YAXSerializableField(DefaultValue = 100)]
        public int NumberOfSteps
        {
            get;
            set;
        }

        [MyBrowsable, Category("Size of output"), YAXSerializableField(DefaultValue = 10)]
        public int SizeOfOutput
        {
            get;
            set;
        }

        [MyBrowsable, Category("Mnist/2DAgnet/PoleBalancing - Pendulum/Gridworld"), YAXSerializableField(DefaultValue = 1)]
        public int Version //1 if it is Mnist
        {
            get;
            set;
        }

        public override void UpdateMemoryBlocks()
        {
            Output.Count = 1;
        }

        public MyIncrementTask IncrementTask { get; private set; }
    }

    /// <summary>
    /// Task that increments all data items by a constant calculated as node's IncrementBase + task's Increment
    /// </summary>
    [Description("Calculate reward")]
    public class MyIncrementTask : MyTask<RewardNode>
    {
        //My2DAgentWorld agentWorld;
        float reward;

        public override void Init(int nGPU)
        {
            //agentWorld = (My2DAgentWorld)Owner.Owner.World;
            reward = 0;
        }

        public override void Execute()
        {
            switch (Owner.Version)
            {
                case 1: // MNIST

                    Owner.InputAgent.SafeCopyToHost();
                    float[] input = Owner.InputAgent.Host;

                    float max = 0f;
                    int label = 0;

                    for (int i = 0; i < Owner.SizeOfOutput; i++)
                    {
                        if (input[i] > max)
                        {
                            max = input[i];
                            label = i;
                        }
                    }
                    if (label == Owner.InputTarget.Host[0])
                    {
                        reward++;
                    }
                    if ((SimulationStep % Owner.NumberOfSteps) == 0 && SimulationStep != 0)
                    {
                        Owner.Output.Host[0] = reward;
                        Owner.Output.SafeCopyToDevice();
                        reward = 0;
                    }
                    break;

                case 2: // 2D agent

                    float agPosX = Owner.InputAgent.Host[0];
                    float agPosY = Owner.InputAgent.Host[1];

                    float tgPosX = Owner.InputTarget.Host[0];
                    float tgPosY = Owner.InputTarget.Host[1];

                    float diff1 = agPosX - tgPosX;
                    float diff2 = agPosY - tgPosY;
                    float dist = (float)Math.Sqrt(Math.Pow(diff1, 2) + Math.Pow(diff2, 2));

                    //how many times reached its goal + inverse of the distance (euclidian, can move in 9 directions
                    if (dist != 0)
                    {
                        reward += (1f / dist);
                    }

                    //CHECK AT EVERY STEP - AGENT REACHED TARGET
                    if (Owner.InputEvent.Host[0] == 1)
                    {
                        reward++;
                    }

                    if (((SimulationStep - 1) % Owner.NumberOfSteps == 0) && SimulationStep != 0)
                    {
                        Owner.Output.Host[0] = reward;
                        Owner.Output.SafeCopyToDevice();
                        reward = 0;
                    }
                    break;

                case 3: // PENDULUM

                    float spherePosX = Owner.InputAgent.Host[0];
                    float spherePosY = Owner.InputAgent.Host[1];
                    float spherePosZ = Owner.InputAgent.Host[2];

                    dist = (float)Math.Sqrt(Math.Pow(spherePosX, 2) + Math.Pow(8 - spherePosY, 2)
                        + Math.Pow(spherePosZ, 2));

                    float norm = (float)Math.Sqrt(128);
                    reward += (norm - dist) / norm;

                    if (((SimulationStep - 1) % Owner.NumberOfSteps == 0) && SimulationStep != 0)
                    {
                        Owner.Output.Host[0] = reward;
                        Owner.Output.SafeCopyToDevice();
                        reward = 0;
                    }
                    break;

                case 4: // gridworld

                    agPosX = Owner.InputTarget.Host[0];
                    agPosY = Owner.InputTarget.Host[1];

                    tgPosX = Owner.InputTarget.Host[6];
                    tgPosY = Owner.InputTarget.Host[7];

                    diff1 = agPosX - tgPosX;
                    diff2 = agPosY - tgPosY;
                    dist = (float)Math.Sqrt(Math.Pow(diff1, 2) + Math.Pow(diff2, 2));

                    //Console.WriteLine("dist " + dist);
                    if (dist > 1)
                    {
                        reward += (1f / dist);
                    }
                    else if (dist >= 0)
                    {
                        reward += 1 - dist;
                    }

                    // if the light is switched to 0 - whatever it means
                    if (Owner.InputTarget.Host[2] == 0 && (SimulationStep - 1) % Owner.NumberOfSteps != 1)
                    {
                        reward += 2;
                    }

                    if (((SimulationStep - 1) % Owner.NumberOfSteps == 0) && SimulationStep != 0)
                    {
                        Owner.Output.Host[0] = reward;
                        //Console.WriteLine("sending reward " + reward);
                        Owner.Output.SafeCopyToDevice();
                        reward = 0;
                    }
                    break;

            
                default:
                    Console.WriteLine("No reward calculation set.");
                    break;
            }
        }
    }
}
