﻿using System;
using System.Collections.Generic;
using System.Text;
using SabberStoneCore.Enums;
using SabberStoneCore.Exceptions;
using SabberStoneCore.Model;
using SabberStoneCore.Model.Entities;
using SabberStoneCore.Model.Zones;
using SabberStoneCore.Tasks.PlayerTasks;
using SabberStoneBasicAI.Meta;

namespace SabberStoneBasicAI.PartialObservation
{
	public partial class POGame
	{
		// define variable for "No Way!" card 
		private static readonly Card PlaceHolder = Cards.FromId("LOEA04_31b");

		private Game game;
		private bool debug;
		private static int max_tries = 10;

		public POGame(Game game, bool debug, Game prevGame = null)
		{
			this.game = game.Clone();
			game.Player1.Game = game;
			game.Player2.Game = game;
			if (prevGame != null)
			{
				hideAdditionalCards(prevGame);
			}
			else
			{
				prepareOpponent();
			}

			this.debug = debug;

			if (debug)
			{
				Console.WriteLine("Game Board");
				Console.WriteLine(game.FullPrint());
			}
		}

		private void prepareOpponent()
		{
			Controller op = game.CurrentOpponent;
			Card placeHolder = PlaceHolder;

			op.DeckCards = DebugDecks.PartialObservationDeck;

			var hand = op.HandZone;
			var span = hand.GetSpan();
			for (int i = span.Length - 1; i >= 0; --i)
			{
				hand.Remove(span[i]);
				game.AuraUpdate();
				hand.Add(Entity.FromCard(in op, in placeHolder));
			}

			var deck = new DeckZone(op);
			for (int i = 0; i < op.DeckZone.Count; i++)
				deck.Add(Entity.FromCard(in op, in placeHolder));
			op.DeckZone = deck;
		}

		private void hideAdditionalCards(Game prevGame)
		{
			//compare current HandZones with the game before and remove all drawn cards
			HandZone prevOp;
			HandZone prevPl;
			if (game.CurrentOpponent.PlayerId == prevGame.CurrentOpponent.PlayerId)
			{
				prevOp = prevGame.CurrentOpponent.HandZone;
				prevPl = prevGame.CurrentPlayer.HandZone;
			}
			else
			{
				prevOp = prevGame.CurrentPlayer.HandZone;
				prevPl = prevGame.CurrentOpponent.HandZone;
			}

			foreach (IPlayable card in game.CurrentOpponent.HandZone)
			{
				if (card.Card.Id != PlaceHolder.Id && !prevOp.Any(x => x.Id == card.Id))
				{
					game.CurrentOpponent.HandZone.Remove(card);
					game.AuraUpdate();
					game.CurrentOpponent.HandZone.Add(Entity.FromCard(game.CurrentOpponent, in PlaceHolder));
				}
			}

			foreach (IPlayable card in game.CurrentPlayer.HandZone)
			{
				if (card.Card.Id != PlaceHolder.Id && !prevPl.Any(x => x.Id == card.Id))
				{
					game.CurrentPlayer.HandZone.Remove(card);
					game.AuraUpdate();
					game.CurrentPlayer.HandZone.Add(Entity.FromCard(game.CurrentPlayer, in PlaceHolder));
				}
			}


		}


		public void addCardToZone(IZone zone, Card card, Controller player)
		{
			var tags = new Dictionary<GameTag, int>();
			tags[GameTag.ENTITY_ID] = game.NextId;
			tags[GameTag.CONTROLLER] = player.PlayerId;
			tags[GameTag.ZONE] = (int)zone.Type;
			IPlayable playable = null;


			switch (card.Type)
			{
				case CardType.MINION:
					playable = new Minion(player, card, tags);
					break;

				case CardType.SPELL:
					playable = new Spell(player, card, tags);
					break;

				case CardType.WEAPON:
					playable = new Weapon(player, card, tags);
					break;

				case CardType.HERO:
					tags[GameTag.ZONE] = (int)Zone.PLAY;
					tags[GameTag.CARDTYPE] = card[GameTag.CARDTYPE];
					playable = new Hero(player, card, tags);
					break;

				case CardType.HERO_POWER:
					tags[GameTag.COST] = card[GameTag.COST];
					tags[GameTag.ZONE] = (int)Zone.PLAY;
					tags[GameTag.CARDTYPE] = card[GameTag.CARDTYPE];
					playable = new HeroPower(player, card, tags);
					break;

				default:
					throw new EntityException($"Couldn't create entity, because of an unknown cardType {card.Type}.");
			}

			zone?.Add(playable);
		}

		public void CreateFullInformationGame(List<Card> deck_player1, DeckZone deckzone_player1, HandZone handzone_player1, List<Card> deck_player2, DeckZone deckzone_player2, HandZone handzone_player2)
		{
			game.Player1.DeckCards = deck_player1;
			game.Player1.DeckZone = deckzone_player1;
			game.Player1.HandZone = handzone_player1;

			game.Player2.DeckCards = deck_player2;
			game.Player2.DeckZone = deckzone_player2;
			game.Player2.HandZone = handzone_player2;
		}

		public void Process(PlayerTask task)
		{
			game.Process(task);
		}

		/**
		 * Simulates the tasks against the current game and
		 * returns a Dictionary with the following POGame-Object
		 * for each task (or null if an exception happened
		 * during that game)
		 */
		public Dictionary<PlayerTask, POGame> Simulate(List<PlayerTask> tasksToSimulate)
		{
			Dictionary<PlayerTask, POGame> simulated = new Dictionary<PlayerTask, POGame>();
			foreach (PlayerTask task in tasksToSimulate)
			{
				bool success = false;
				if (!(task.HasSource && task.Source.Card.Id == PlaceHolder.Id))
				{
					for (int tries = 0; tries < max_tries; tries++)
					{
						try
						{
							Game clone = game.Clone();
							clone.Process(task);
							simulated.Add(task, new POGame(clone, this.debug, game));
							success = true;
							break;
						}
						catch (Exception e)
						{
							Console.Write(e);
							//Console.WriteLine("Failed to copy");
						}
					}
					if (!success)
					{
						simulated.Add(task, null);
						//Console.WriteLine("Failed to copy very often");
					}
				}
				else
				{
					simulated.Add(task, null);
					//Console.WriteLine("Agent tries to play debug card!");
				}
			}
			return simulated;

		}

		public POGame getCopy(bool? debug = null)
		{
			return new POGame(game, debug ?? this.debug);
		}

		public Game getGame()
		{
			return game;
		}


		public string FullPrint()
		{
			return game.FullPrint();
		}

		public string PartialPrint()
		{
			var str = new StringBuilder();
			if (game.CurrentPlayer == game.Player1)
			{
				str.AppendLine(game.Player1.HandZone.FullPrint());
				str.AppendLine(game.Player1.Hero.FullPrint());
				str.AppendLine(game.Player1.BoardZone.FullPrint());
				str.AppendLine(game.Player2.BoardZone.FullPrint());
				str.AppendLine(game.Player2.Hero.FullPrint());
				str.AppendLine(String.Format("Opponent Hand Cards: {0}", game.Player2.HandZone.Count));
			}
			if (game.CurrentPlayer == game.Player2)
			{
				str.AppendLine(String.Format("Opponent Hand Cards: {0}", game.Player1.HandZone.Count));
				str.AppendLine(game.Player1.Hero.FullPrint());
				str.AppendLine(game.Player1.BoardZone.FullPrint());
				str.AppendLine(game.Player2.BoardZone.FullPrint());
				str.AppendLine(game.Player2.Hero.FullPrint());
				str.AppendLine(game.Player2.HandZone.FullPrint());
			}

			return str.ToString();
		}


	}

	/// <summary>
	/// Standard Getters for the current game
	/// </summary>
	partial class POGame
	{

		/// <summary>
		/// Gets or sets the turn count.
		/// </summary>
		/// <value>The amount of player turns that happened in the game. When the game starts (after Mulligan),
		/// value will equal 1.</value>
		public int Turn
		{
			get { return game[GameTag.TURN]; }
		}

		/// <summary>
		/// Gets or sets the game state.
		/// </summary>
		/// <value><see cref="State"/></value>
		public State State
		{
			get { return (State)game[GameTag.STATE]; }
		}

		/// <summary>
		/// Gets or sets the first card played this turn.
		/// </summary>
		/// <value>The entityID of the card.</value>
		public int FirstCardPlayedThisTurn
		{
			get { return game[GameTag.FIRST_CARD_PLAYED_THIS_TURN]; }
		}

		/// <summary>
		/// The controller which goes 'first'. This player's turn starts after Mulligan.
		/// </summary>
		/// <value><see cref="Controller"/></value>
		public Controller FirstPlayer
		{
			get
			{
				return game.Player1[GameTag.FIRST_PLAYER] == 1 ? game.Player1 : game.Player2[GameTag.FIRST_PLAYER] == 1 ? game.Player2 : null;
			}
		}

		/// <summary>
		/// Gets or sets the controller delegating the current turn.
		/// Thanks to Milva
		/// </summary>
		/// <value><see cref="Controller"/></value>
		public Controller CurrentPlayer => game.CurrentPlayer;


		/// <summary>
		/// Gets the opponent controller of <see cref="CurrentPlayer"/>.
		/// </summary>
		/// <value><see cref="Controller"/></value>
		public Controller CurrentOpponent => game.CurrentOpponent;

		/// <summary>
		/// Gets or sets the CURRENT step. These steps occur within <see cref="State.RUNNING"/> and
		/// indicate states which are used to process actions.
		/// </summary>
		/// <value><see cref="Step"/></value>
		public Step Step
		{
			//get { return (Step)this[GameTag.STEP]; }
			//set { this[GameTag.STEP] = (int)value; }
			get { return (Step)game.GetNativeGameTag(GameTag.STEP); }
		}

		/// <summary>
		/// Gets or sets the NEXT step. <seealso cref="Step"/>
		/// </summary>
		/// <value><see cref="Step"/></value>
		public Step NextStep
		{
			get { return (Step)game.GetNativeGameTag(GameTag.NEXT_STEP); }
		}

		/// <summary>
		/// Gets or sets the number of killed minions for this turn.
		/// </summary>
		/// <value>The amount of killed minions.</value>
		public int NumMinionsKilledThisTurn
		{
			get { return game[GameTag.NUM_MINIONS_KILLED_THIS_TURN]; }
		}

		/// <summary>Gets the heroes.</summary>
		/// <value><see cref="Hero"/></value>
		public List<Hero> Heroes => game.Heroes;

		/// <summary>Gets ALL minions (from both sides of the board).</summary>
		/// <value><see cref="Minion"/></value>
		public List<Minion> Minions => game.Minions;

		/// <summary>Gets ALL characters.</summary>
		/// <value><see cref="ICharacter"/></value>
		public List<ICharacter> Characters => game.Characters;
	}
}
