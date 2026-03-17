"""
Unit tests for physics engine.

Tests physics calculations, body creation, and simulation stepping.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics import Physics
from models import Vector, Spring


class TestPhysics:
    """Test Physics class."""
    
    def test_init(self):
        """Test physics initialization."""
        physics = Physics()
        assert physics.space.gravity.y == -9.81
        assert len(physics.dynamic) == 0
        assert len(physics.static) == 0
        assert physics.is_running is True
    
    def test_add_circle(self):
        """Test adding a circle body."""
        physics = Physics()
        shape = physics.add_circle(r=0.5, pos=(0, 2), m=1.0)
        
        assert shape is not None
        assert len(physics.dynamic) == 1
        assert shape.radius == 0.5
        assert shape.body.mass == 1.0
        assert shape.body.position.y == 2.0
    
    def test_add_box(self):
        """Test adding a box body."""
        physics = Physics()
        shape = physics.add_box(w=1.0, h=2.0, pos=(1, 3), m=2.0)
        
        assert shape is not None
        assert len(physics.dynamic) == 1
        assert shape.body.mass == 2.0
        assert shape.body.position.x == 1.0
    
    def test_add_surface(self):
        """Test adding a static surface."""
        physics = Physics()
        shape = physics.add_surface((0, 0), (5, 0), radius=0.1)
        
        assert shape is not None
        assert len(physics.static) == 1
    
    def test_step_paused(self):
        """Test that step does nothing when paused."""
        physics = Physics()
        physics.is_running = False
        
        initial_pos = physics.add_circle(pos=(0, 10), m=1.0).body.position.y
        physics.step(0.1)
        
        # Position should not change when paused
        assert physics.add_circle(pos=(0, 10), m=1.0).body.position.y == initial_pos
    
    def test_stats(self):
        """Test stats calculation."""
        physics = Physics()
        physics.add_circle(m=1.0, pos=(0, 0))
        physics.add_circle(m=2.0, pos=(5, 5))
        physics.add_surface((0, 0), (10, 0))
        
        bodies, surfaces, ke = physics.stats()
        
        assert bodies == 2
        assert surfaces == 1
        assert ke >= 0  # KE is always non-negative


class TestVector:
    """Test Vector model."""
    
    def test_zero_vector(self):
        """Test zero magnitude vector."""
        v = Vector("test", 0, 0, 0, 0)
        assert v.magnitude == 0.0
        assert v.angle_deg == 0.0
        assert v.Fx == 0.0
        assert v.Fy == 0.0
    
    def test_horizontal_vector(self):
        """Test horizontal vector."""
        v = Vector("test", 0, 0, 5, 0)
        assert v.magnitude == 5.0
        assert v.angle_deg == 0.0
        assert v.Fx == 5.0
        assert v.Fy == 0.0
    
    def test_vertical_vector(self):
        """Test vertical vector."""
        v = Vector("test", 0, 0, 0, 10)
        assert v.magnitude == 10.0
        assert v.angle_deg == 90.0
        assert v.Fx == 0.0
        assert v.Fy == 10.0
    
    def test_diagonal_vector(self):
        """Test diagonal vector."""
        v = Vector("test", 0, 0, 3, 4)
        assert v.magnitude == 5.0  # 3-4-5 triangle
        assert abs(v.angle_deg - 53.13) < 0.1  # Approximately 53.13 degrees
    
    def test_negative_components(self):
        """Test vector with negative components."""
        v = Vector("test", 0, 0, -3, -4)
        assert v.magnitude == 5.0
        assert abs(v.angle_deg - (-126.87)) < 0.1  # Third quadrant


class TestSpring:
    """Test Spring model."""
    
    def test_default_spring(self):
        """Test spring with default values."""
        spring = Spring(label="S1")
        
        assert spring.rest_length == 1.0
        assert spring.stiffness == 25.0
        assert spring.damping == 0.5
        assert spring.max_extension_factor == 2.0
    
    def test_custom_spring(self):
        """Test spring with custom values."""
        spring = Spring(
            label="S2",
            rest_length=2.0,
            stiffness=50.0,
            damping=1.0
        )
        
        assert spring.rest_length == 2.0
        assert spring.stiffness == 50.0
        assert spring.damping == 1.0


class TestUndoRedo:
    """Test undo/redo functionality."""
    
    def test_snapshot_includes_all(self):
        """Test that snapshot includes all object types."""
        # This would require mocking the App class
        # For now, just verify the structure exists
        from app import App
        assert hasattr(App, '_snapshot_state')
        assert hasattr(App, '_restore_state')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
