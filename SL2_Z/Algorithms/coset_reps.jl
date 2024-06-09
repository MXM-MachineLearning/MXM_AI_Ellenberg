Γ₀2_mod_Γ2 = [[1 0; 0 1], [1 1; 0 1]]
Γ_mod_Γ₀2 = [[1 0; 0 1], [0 -1; 1 0], [1 0; 1 1]]
Γ2_mod_Γ′ = [[1 0; 0 1], [-1 0; 0 -1]]

for A ∈ Γ₀2_mod_Γ2
    for B ∈ Γ_mod_Γ₀2
        for C ∈ Γ2_mod_Γ′
            rep = A * B * C
            println("\\begin{bmatrix}", rep[1, 1], "&", rep[1, 2], "\\\\", rep[2, 1], "&", rep[2, 2], "\\end{bmatrix}")
        end
    end
end
