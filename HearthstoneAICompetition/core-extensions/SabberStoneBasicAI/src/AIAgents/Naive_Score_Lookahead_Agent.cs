using System;
using System.Collections.Generic;
using SabberStoneCore.Tasks.PlayerTasks;
using SabberStoneBasicAI.PartialObservation;
using SabberStoneCore.Enums;
using System.Linq;
using SabberStoneBasicAI.Score;
using SabberStoneCore.Model.Entities;

//Submission for Master
namespace SabberStoneBasicAI.AIAgents.BetterGreedyBot
{

	class MyAgentSebastianMiller2 : AbstractAgent
	{
		public override void FinalizeAgent()
		{
		}

		public override void FinalizeGame()
		{
		}

		public override PlayerTask GetMove(POGame game)
		{
			var player = game.CurrentPlayer;
			var validOpts = game.Simulate(player.Options()).Where(x => x.Value != null);
			var optcount = validOpts.Count();

			var returnValue = validOpts.Any() ?
				validOpts.Select(x => score(x, player.PlayerId, (optcount >= 5) ? ((optcount >= 25) ? 1 : 2) : 3)).OrderBy(x => x.Value).Last().Key :
				player.Options().First(x => x.PlayerTaskType == PlayerTaskType.END_TURN);

			return returnValue;

			KeyValuePair<PlayerTask, int> score(KeyValuePair<PlayerTask, POGame> state, int player_id, int max_depth = 3)
			{
				int max_score = int.MinValue;
				if (max_depth > 0 && state.Value.CurrentPlayer.PlayerId == player_id)
				{
					var subactions = state.Value.Simulate(state.Value.CurrentPlayer.Options()).Where(x => x.Value != null);

					foreach (var subaction in subactions)
						max_score = Math.Max(max_score, score(subaction, player_id, max_depth - 1).Value);


				}
				max_score = Math.Max(max_score, Score(state.Value, player_id));
				return new KeyValuePair<PlayerTask, int>(state.Key, max_score);
			}
		}

		private static int Score(POGame state, int playerId)
		{
			var p = state.CurrentPlayer.PlayerId == playerId ? state.CurrentPlayer : state.CurrentOpponent;
			return new BetterScore { Controller = p }.Rate();
		}

		public override void InitializeAgent()
		{
		}

		public override void InitializeGame()
		{
		}
	}
}
