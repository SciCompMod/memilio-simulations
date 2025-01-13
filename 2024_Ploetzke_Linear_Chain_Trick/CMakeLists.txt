
add_executable(lct_impact_distribution_assumption lct_impact_distribution_assumption.cpp)

# Link to right memilio project.
target_link_libraries(lct_impact_distribution_assumption PRIVATE memilio ode_sir)
