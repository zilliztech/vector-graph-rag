package com.zjkl.vectorgraphrag.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Objects;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class Triplet {
    private String subject;
    private String predicate;
    private String object;

    public String toRelationText() {
        return subject + " " + predicate + " " + object;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Triplet triplet)) return false;
        return subject.equalsIgnoreCase(triplet.subject)
                && predicate.equalsIgnoreCase(triplet.predicate)
                && object.equalsIgnoreCase(triplet.object);
    }

    @Override
    public int hashCode() {
        return Objects.hash(
                subject.toLowerCase(),
                predicate.toLowerCase(),
                object.toLowerCase()
        );
    }
}
