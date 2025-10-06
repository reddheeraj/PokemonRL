"""
Memory reading utilities for Pokemon Red
These help extract game state information from the emulator
"""

class PokemonRedMemory:
    """Helper class to read and interpret Pokemon Red game memory"""
    
    def __init__(self, pyboy):
        self.pyboy = pyboy
        
    def read_byte(self, address):
        """Read a single byte from memory"""
        # PyBoy 2.x API: use memory[address] instead of get_memory_value()
        return self.pyboy.memory[address]
    
    def read_word(self, address):
        """Read a 16-bit word from memory (little endian)"""
        # PyBoy 2.x API: use memory[address] instead of get_memory_value()
        low = self.pyboy.memory[address]
        high = self.pyboy.memory[address + 1]
        return (high << 8) | low
    
    def get_player_position(self):
        """Get player X, Y coordinates"""
        # These addresses are for Pokemon Red
        x = self.read_byte(0xD362)
        y = self.read_byte(0xD361)
        return x, y
    
    def get_map_id(self):
        """Get current map ID"""
        return self.read_byte(0xD35E)
    
    def get_party_count(self):
        """Get number of Pokemon in party"""
        return self.read_byte(0xD163)
    
    def is_in_battle(self):
        """Check if currently in a battle"""
        battle_type = self.read_byte(0xD057)
        return battle_type != 0
    
    def get_battle_type(self):
        """Get the type of battle (0=none, 1=wild, 2=trainer)"""
        return self.read_byte(0xD057)
    
    def get_first_pokemon_hp(self):
        """Get HP of first Pokemon in party"""
        hp_current = self.read_word(0xD16C)
        return hp_current
    
    def get_first_pokemon_max_hp(self):
        """Get max HP of first Pokemon in party"""
        hp_max = self.read_word(0xD18D)
        return hp_max
    
    def in_grass_area(self):
        """Heuristic to check if player is likely in grass area"""
        map_id = self.get_map_id()
        # Route 1 has ID around 12-13, but this may vary
        # We'll consider maps adjacent to Pallet Town
        return map_id in [12, 13, 33, 37]
    
    def get_game_state_hash(self):
        """Create a hash of the current game state for novelty detection"""
        x, y = self.get_player_position()
        map_id = self.get_map_id()
        party_count = self.get_party_count()
        in_battle = self.is_in_battle()
        
        # Simple hash combining position, map, and game progress
        state_hash = (map_id << 16) | (x << 8) | y | (party_count << 24) | (in_battle << 31)
        return state_hash

